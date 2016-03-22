#!/usr/bin/env python
# coding=utf-8
"""
 * This file is part of the Aietes Framework (https://github.com/andrewbolster/aietes)
 *
 * (C) Copyright 2013 Andrew Bolster (http://andrewbolster.info/) and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Andrew Bolster, Queen's University Belfast (-July 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import logging
from collections import namedtuple
from itertools import product
from operator import attrgetter

import numpy as np

from aietes.Tools import MapEntry, distance, fudge_normal, DEBUG, unit, mag, listfix, sixvec, spherical_distance, \
    ConfigError, angle_between, random_three_vector, random_xy_vector, agitate_position


# DEBUG = False


class BasicWaypoint(object):
    position = None
    prox = None

    def __init__(self, position, prox):
        self.position = position
        self.prox = prox

    def __repr__(self):
        return "({0},{1})".format(self.position, self.prox)


waypoint = namedtuple("waypoint", ['position', 'prox'])


class Behaviour(object):
    """
    Generic Representation of a Nodes behavioural characteristics
    #TODO should be a state machine?
    """

    def __init__(self, *args, **kwargs):
        # TODO internal representation of the environment
        self.node = kwargs.get('node')
        self.bev_config = kwargs.get('bev_config')
        self.map = kwargs.get('map', None)
        self.debug = DEBUG and self.node.debug
        self._start_log(self.node)
        if self.debug:
            self.logger.debug('from bev_config: {0!s}'.format(self.bev_config))
            # self.logger.setLevel(logging.DEBUG)
        self.update_rate = 1
        self.memory = {}
        self.behaviours = [self.avoid_wall]
        self.simulation = self.node.simulation
        self.env_shape = np.asarray(self.simulation.environment.shape)
        self.neighbours = {}
        self.nearest_neighbours = []
        self.n_nearest_neighbours = listfix(
            int, self.bev_config['nearest_neighbours'])
        self.neighbourhood_max_rad = listfix(
            int, self.bev_config['neighbourhood_max_rad'])
        self.neighbour_min_rad = listfix(
            int, self.bev_config['neighbourhood_min_rad'])
        self.positional_accuracy = listfix(
            float, self.bev_config['positional_accuracy'])
        self.horizon = 200
        # Getting lazy towards the end...
        self.wallCheckDisabled = True

    def _start_log(self, parent):
        self.logger = parent.logger.getChild(
            "Bev:{0!s}".format(self.__class__.__name__))

        self.logger.debug("Launched Behaviour")

    @staticmethod
    def normalize_behaviour(force_vector):
        return force_vector

    def neighbour_map(self):
        """
        Returns a filtered map pulled from the map excluding self
        """
        orig_map = dict((k, v)
                        for k, v in self.map.items() if v.object_id != self.node.id)

        # Internal Map of the neighbourhood based on best-guess of location
        # (circa 5m / 5ms)
        for k, v in orig_map.items():
            orig_map[k].position = fudge_normal(
                v.position, self.positional_accuracy)
            orig_map[k].velocity = fudge_normal(
                v.velocity, self.positional_accuracy)

        return orig_map

    def add_memory(self, object_id, position, velocity):
        """
        Called by node lifecycle to update the internal representation of the environment
        :param object_id:
        :param position:
        :param velocity:
        """
        # TODO expand this to do SLAM?
        self.memory += MapEntry(object_id, position, velocity)

    def process(self):
        """
        Process current map and memory information and update velocities
        """
        self.debug = self.debug and self.node == self.node.fleet.nodes[0]
        self.neighbours = self.neighbour_map()
        self.nearest_neighbours = self.get_nearest_neighbours(n_neighbours=self.n_nearest_neighbours)
        contributions = {}
        force_vector = np.zeros(3)

        for behaviour in self.behaviours:
            try:
                contributions[behaviour.__name__] = behaviour(
                    self.node.position, self.node.velocity)
            except Exception as exp:
                self.logger.error(
                    "{0!s}({1!s},{2!s})".format(behaviour.__name__, self.node.position, self.node.velocity))
                raise

        force_vector += sum(contributions.values())

        # TODO Under Drift, it's probably better to do wall-detection twice: once on node and once on environment
        # force_vector = self.avoid_wall(self.node.get_pos(), self.node.velocity, force_vector)
        if self.debug and DEBUG:
            self.logger.debug("Response:{0!s}".format(force_vector))
        if self.debug and DEBUG:
            total = sum(map(mag, contributions.values()))
            if total > 0:
                self.logger.debug("contributions: {0!s} of {1:3f}".format(
                    [
                        "{0!s}:{1:f}%".format(func, 100 * mag(value) / total)
                        for func, value in contributions.iteritems()
                        ], total)
                                  )
            else:
                self.logger.debug("contributions: None")
        if self.debug and DEBUG:
            total = sum(map(mag, contributions.values()))
            for func, value in contributions.iteritems():
                self.logger.debug(
                    "{0!s}:{1:.2f}:{2!s}".format(func, 100 * mag(value) / total, sixvec(value)))
        #force_vector = fudge_normal(force_vector, 0.012)  # Random factor
        self.node.push(force_vector, contributions=contributions)
        return

    def get_nearest_neighbours(self, position=None, n_neighbours=None):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        :param position:
        :param n_neighbours:
        """

        if position is None:
            target_pos = self.node.position
        else:
            target_pos = position
        # Sort and filter Neighbours from self.map by distance
        neighbours_with_distance = [MapEntry(key,
                                             value.position, value.velocity,
                                             name=self.simulation.reverse_node_lookup(
                                                 key).name,
                                             distance=distance(target_pos, value.position)

                                             ) for key, value in self.neighbours.items()]
        # self.logger.DEBUG("Got Distances: %s"%neighbours_with_distance)
        nearest_neighbours = sorted(neighbours_with_distance, key=attrgetter('distance')
                                    )
        # Select N neighbours in order
        # self.logger.DEBUG("Nearest Neighbours:%s"%nearest_neighbours)
        if n_neighbours is not None:
            nearest_neighbours = nearest_neighbours[:n_neighbours]
        return nearest_neighbours

    def repulse_from_position(self, position, repulsive_position, d_limit=1):
        """

        :param position:
        :param repulsive_position:
        :param d_limit:
        :return: :raise RuntimeError:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        distance_val = distance(position, repulsive_position)
        force_vector = unit(position - repulsive_position) * \
                       d_limit / float(min(distance_val, self.neighbourhood_max_rad))

        if distance_val < 1:
            raise RuntimeError("Too close to {0!s} ({1!s}) moving at {2!s}; I was at {3!s} moving at {4!s}".format(self.get_nearest_neighbours(repulsive_position)[0].name,
                                self.get_nearest_neighbours(
                                    repulsive_position)[0].position,
                                sixvec(
                                    self.get_nearest_neighbours(repulsive_position)[0].velocity),
                                self.node.position,
                                sixvec(self.node.velocity)
                                ))
        if self.debug:
            self.logger.debug(
                "Repulsion from {0!s}: {1!s}, at range of {2!s}".format(force_vector, repulsive_position, distance_val))
        return force_vector

    def attract_to_position(self, position, attractive_position, d_limit=1):
        """

        :param position:
        :param attractive_position:
        :param d_limit:
        :return:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        distance_val = distance(position, attractive_position)
        force_vector = unit(attractive_position - position) * \
                       (min(distance_val, self.neighbourhood_max_rad) / float(d_limit))
        if self.debug:
            self.logger.debug(
                "Attraction to {0!s}: {1!s}, at range of {2!s}".format(force_vector, attractive_position, distance_val))
        return force_vector

    def avoid_wall(self, position, velocity):
        """
        Called by responseVector to avoid walls to a distance of half min distance
        :param position:
        :param velocity:
        """
        response = np.array([0, 0, 0], dtype=np.float)
        min_dist = self.neighbourhood_max_rad / 2.0
        avoid = False
        avoiding_position = position.copy()
        if np.any((np.zeros(3) + min_dist) > position):
            if self.debug:
                self.logger.debug(
                    "Too Close to the Origin-surfaces: {0!s}".format(position))
            offending_dim = position.argmin()
            avoiding_position[offending_dim] = float(0.0)
            avoid = True

        if np.any(position > (self.env_shape - min_dist)):
            if self.debug:
                self.logger.debug(
                    "Too Close to the Upper-surfaces: {0!s}".format(position))
            offending_dim = position.argmax()
            avoiding_position[offending_dim] = float(
                self.env_shape[offending_dim])
            avoid = True

        if avoid:
            # response = 0.5 * (position-avoiding_position)
            self.debug = True
            try:
                repulse = self.repulse_from_position(
                    position, avoiding_position, 2 * min_dist)
                # Reflect rather than just bounce; response is the normal vector of the surface
                r_mag = mag(repulse)
                # If we're at too narrow an angle we'll get stuck on a wall
                angle = angle_between(repulse, -velocity)
                if angle < 2.0 * np.pi / 5.0:  # 72 degrees of wall norm
                    response = repulse + velocity - (2 * (np.dot(velocity, repulse)) / (r_mag ** 2)) * repulse
                else:
                    response = repulse
            except RuntimeError:
                raise RuntimeError(
                    "Crashed out of environment with given position:{0}, wall position:{1}".format(position,
                                                                                                 avoiding_position))
                # response = (avoiding_position-position)
            self.logger.debug("Wall Avoidance:{0!s}".format(response))
            if hasattr(self, 'my_direction') and not hasattr(self, 'time_travelled'):
                # Something planned to go this way, lets stop that and hope it chooses a better direction
                self.my_direction = unit(response)
            if hasattr(self, 'time_travelled'):
                # Is something that cares about timing watching? Because then only register on the
                # first 'avoid' (which will have the right direction)
                if self.time_travelled > 10:
                    self.time_travelled = 0
                    self.my_direction = unit(response)

        return response


class RandomWalk(Behaviour):
    """
    Generic Wandering Behaviour
    """

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        self.behaviours.append(self.random_walk)
        self.wallCheckDisabled = True
        self.my_random_direction = random_three_vector()

    def random_walk(self, position, velocity):
        # Roughly 6 degrees or pi/32 rad
        """

        :param position:
        :param velocity:
        :return:
        """
        if angle_between(velocity, self.my_random_direction) < 0.2:
            self.my_random_direction = random_three_vector()
        return np.asarray(self.my_random_direction)


class RandomFlatWalk(Behaviour):
    """
    Generic Wandering Behaviour on a plane
    travels for about a 5th of the size of the environment before turning
    """

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        self.behaviours.append(self.random_walk)
        self.wallCheckDisabled = False
        self.my_direction = random_xy_vector()
        self.time_travelled = 0
        self._time_limit = self.node.simulation.environment.shape[0] / 5

    def random_walk(self, position, velocity):
        """

        Pick a random direction, get to roughly 6 degrees or pi/32 rad
        of that vector, and continue for a given timelimit

        :param position:
        :param velocity:
        :return:
        """

        if angle_between(velocity, self.my_direction) < 0.2:
            if self.time_travelled > self._time_limit:
                self.time_travelled = 0
                self.my_direction = random_xy_vector()
            else:
                self.time_travelled += 1

        return np.asarray(self.my_direction)


class RandomFlatCentredWalk(RandomFlatWalk):
    # Should probably add some intelligence in here to select a random point within the environment to head towards
    # rather than picking a random direction....

    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        super(RandomFlatCentredWalk, self).__init__(*args, **kwargs)
        self.wallCheckDisabled = False
        self.original_position = self.node.get_pos()
        self.behaviours.append(self.attract_to_origin)

    def attract_to_origin(self, position, velocity):
        """

        :param position:
        :param velocity:
        :return:
        """
        return np.asarray(self.attract_to_position(position, self.original_position)) * 0.001


class Nothing(Behaviour):
    """
    Do Nothing
    """

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)


class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        self.clumping_factor = self.bev_config['clumping_factor']
        self.repulsive_factor = listfix(
            float, self.bev_config['repulsive_factor'])
        self.schooling_factor = listfix(
            float, self.bev_config['schooling_factor'])
        self.collision_avoidance_d = listfix(
            float, self.bev_config['collision_avoidance_d'])

        self.behaviours.append(self.clumping_vector)
        self.behaviours.append(self.repulsive_vector)
        self.behaviours.append(self.local_heading)

        assert self.n_nearest_neighbours > 0
        assert self.neighbourhood_max_rad > 0
        assert self.neighbour_min_rad > 0

    def clumping_vector(self, position, velocity):
        """


        :param position:
        :param velocity:
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector = np.array([0, 0, 0], dtype=np.float)
        if len(self.nearest_neighbours) < 1:
            raise RuntimeError("I don't have any neighbours!")

        for neighbour in self.nearest_neighbours:
            vector += np.array(neighbour.position)

        try:
            # This assumes that the map contains one entry for each non-self
            # node
            self.neighbourhood_com = vector / len(self.nearest_neighbours)
            if self.debug:
                self.logger.debug("Cluster Centre,position,factor,neighbours: {0!s},{1!s},{2!s},{3!s}".format(
                    self.neighbourhood_com, vector, self.clumping_factor, len(self.nearest_neighbours)))
                # Return the fudged, relative vector to the centre of the
                # cluster
            force_vector = self.attract_to_position(
                position, self.neighbourhood_com, self.collision_avoidance_d)
        except ZeroDivisionError:
            self.logger.error("Zero Division Error: Returning zero vector")
            force_vector = position - position
        except FloatingPointError:
            self.logger.error("FPE: vector={0!s}".format(str(vector)))
            raise

        if self.debug:
            self.logger.debug("Clump:{0!s}".format(force_vector))
        return self.normalize_behaviour(force_vector) * self.clumping_factor

    def repulsive_vector(self, position, velocity):
        """


        :param position:
        :param velocity:
        Represents the Short Range Repulsion behaviour:
            Steer away from it based on a repulsive desire curve
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            if distance(position, neighbour.position) < self.collision_avoidance_d:
                part_vector = self.repulse_from_position(
                    position, neighbour.position, self.collision_avoidance_d)
                if self.debug:
                    self.logger.debug(
                        "Avoiding {0!s}:{1:f}:{2!s}".format(
                            neighbour.name, distance(position, neighbour.position), sixvec(part_vector)))
                force_vector += part_vector

                # Return an inverse vector to the obstacles
        if self.debug and not mag(force_vector) > 0.0:
            self.logger.debug("Repulse:{0!s}".format(force_vector))
        return self.normalize_behaviour(force_vector) * self.repulsive_factor

    def local_heading(self, position, velocity):
        """
        Represents Local Average Heading
        :param position:
        :param velocity:
        """
        vector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            vector += fudge_normal(unit(neighbour.velocity),
                                   max(abs(unit(neighbour.velocity))) / 3)
        force_vector = (vector / (len(self.nearest_neighbours)))
        if self.debug:
            self.logger.debug("Schooling:{0!s}".format(force_vector))
        if self.debug:
            self.logger.debug("V:{0!s},F:{1!s}, {2:f}".format(
                unit(velocity), unit(force_vector), spherical_distance(unit(velocity), unit(force_vector))))
        d = spherical_distance(unit(velocity), unit(force_vector))
        return self.normalize_behaviour(force_vector) * self.schooling_factor

    def _percieved_vector(self, node_id, time):
        """
        Finite Difference Estimation
        from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
        """
        node_history = sorted(
            filter(lambda x: x.object_id == node_id, self.memory), key=time)
        return (node_history[-1].position - node_history[-2].position) / (node_history[-1].time - node_history[-2].time)


class AlternativeFlockMixin(object):
    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        self.behaviours.remove(self.clumpingVector)
        self.behaviours.remove(self.repulsiveVector)
        self.behaviours.append(self.potential_vector)

    def potential_vector(self, position, velocity):
        """
        Computed singular collision/formation control vector instead of joint clumping/repulsive vectors
        :param position:
        :param velocity:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            d = distance(position, neighbour.position)
            v = unit(neighbour.position - position)  # Direction to go
            if d < self.collision_avoidance_d:
                v = -v
            # force to go
            f = np.log(d) + (self.collision_avoidance_d / float(d))
            nforce_vector = f * v
            if self.debug:
                self.logger.debug(
                    "PotentialV: {0!s}, for D:{1!s}, F:{2!s}, V:{3!s}".format(nforce_vector, d, f, v))
            force_vector += nforce_vector
        return force_vector * self.clumping_factor


class AlternativeFlock(Flock, AlternativeFlockMixin):
    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        AlternativeFlockMixin.__init__(self, *args, **kwargs)


class WaypointMixin(object):
    """
    Waypoint MixIn Class defines the general waypoint behaviour and includes the inner 'waypoint' object class.
    """

    def __init__(self, *args, **kwargs):
        self.waypoint_factor = listfix(float, self.bev_config['waypoint_factor'])
        self.waypoints = []
        self.nextwaypoint = None
        self.waypointloop = True
        self._nwpc = 0
        self.behaviours.append(self.waypoint_vector)

    def activate(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :raise ConfigError:
        """
        if not hasattr(self, str(self.bev_config['waypoint_style'])):
            raise ConfigError(
                "Cannot generate using waypoint definition:{0!s}".format(self.bev_config['waypoint_style']))
        else:
            generator = attrgetter(str(self.bev_config['waypoint_style']))
            g = generator(self)
            if self.debug:
                self.logger.debug("Generating waypoints: {0!s}".format(g.__name__))
            g()

    def patrol_cube(self):
        """
        Generates a cubic patrol loop within the environment
        """
        shape = np.asarray(self.env_shape)
        prox = 50
        cubedef = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        )
        self.waypoints = [
            waypoint(shape * (((vertex - 0.5) / 3) + 0.5), prox) for vertex in cubedef]
        self.nextwaypoint = 0

    def there_and_back_again(self):
        """
        Generates a linear patrol path, assumed to be from the centre of the environment, to an x-dimension face, and back to centre
        """
        shape = np.asarray(self.env_shape)
        prox = 50
        self.waypoints = [waypoint(shape * (((vertex - 0.5) / 3) + 0.5), prox)
                          for vertex in np.asarray([[0.5, 0, 0.5], [0.5, 0.5, 0.5]])]
        self.nextwaypoint = 0

    def waypoint_vector(self, position, velocity):
        """

        :param position:
        :param velocity:
        :return:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        if self.nextwaypoint is not None:
            if isinstance(self, SoloWaypoint) or len(self.nearest_neighbours) == 0:
                neighbourhood_avg = position
            else:
                neighbourhood_avg = sum(
                    n.position for n in self.nearest_neighbours) / len(self.nearest_neighbours)

            target = self.waypoints[self.nextwaypoint]
            real_d = distance(position, target.position)
            neighbourhood_d = distance(neighbourhood_avg, target.position)
            if neighbourhood_d < target.prox or real_d < target.prox:
                self.goto_next_waypoint(target.position, real_d)
            else:
                force_vector = self.attract_to_position(
                    position, target.position, target.prox / 2)
        else:
            if self.node.on_mission:
                self.logger.info("No Waypoint")

        return self.normalize_behaviour(force_vector) * self.waypoint_factor

    def goto_next_waypoint(self, position, real_d):
        # GRANT ACHIEVEMENT
        """

        :param position:
        :param real_d:
        """
        self.node.grant_achievement((position, real_d))
        self.nextwaypoint = (self.nextwaypoint + 1)

        if self.nextwaypoint > len(self.waypoints) - 1:
            if self.waypointloop:
                self.nextwaypoint = 0
            else:
                self.node.mission_accomplished()
                self.nextwaypoint = None


class Waypoint(Flock, WaypointMixin):
    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        WaypointMixin.__init__(self, *args, **kwargs)


class AlternativeWaypoint(AlternativeFlock, Waypoint):
    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        AlternativeFlock.__init__(self, *args, **kwargs)
        WaypointMixin.__init__(self, *args, **kwargs)


class SoloWaypoint(Nothing, WaypointMixin):
    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        Nothing.__init__(self, *args, **kwargs)
        WaypointMixin.__init__(self, *args, **kwargs)


class StationKeep(Nothing):
    """
    Simple behaviour that represents a station-keeping state / buoy.
    """

    def __init__(self, *args, **kwargs):
        Nothing.__init__(self, *args, **kwargs)


class FleetLawnmower(Flock, WaypointMixin):
    """
    Repeating Lawnmower Behaviour across a 2D slice of the environment extent
        Subdivides environment into N-Overlapping patterns based on fleet size

    Assuming a swatch width of 250m for the time being.
    Turning is not controlled in detail, however in practice, it is assumed that
    the pinning waypoints would be in overlapping regions rather than being
    destructively lost.

    Uses the repulsive behaviour from Flock and uses the same repulsive
    factor config entry, but disregards other behaviours
    """

    # TODO The guts of this are in notebook

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        WaypointMixin.__init__(self, *args, **kwargs)
        self.waypointloop = False
        self.n_nearest_neighbours = listfix(
            int, self.bev_config['nearest_neighbours'])
        self.repulsive_factor = listfix(
            float, self.bev_config['repulsive_factor'])
        self.collision_avoidance_d = listfix(
            float, self.bev_config['collision_avoidance_d'])

        self.behaviours.append(self.boresight)
        self.behaviours.append(self.tracked_avoidance)

        assert self.n_nearest_neighbours > 0
        assert self.neighbourhood_max_rad > 0
        assert self.neighbour_min_rad > 0

    def activate(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        if self.debug:
            self.logger.debug("Generating waypoints: Lawnmower")
        self.lawnmower(self.node.fleet.node_count(), overlap=30, twister=False)

    def boresight(self, position, velocity):
        """

        :param position:
        :param velocity:
        :return:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        # We only care about boresight tracking while we're in survey
        careaboutboresight = self.nextwaypoint is not None and self.node.on_mission
        if careaboutboresight:
            target = self.waypoints[self.nextwaypoint]
            angle = angle_between(target.position - position, velocity)
            if self.debug:
                self.logger.info("Boresight:{0}@{1}".format(
                    np.rad2deg(angle), distance(target.position, position)))
            force_vector = -velocity * np.pi * angle * self.waypoint_factor
        return force_vector

    def tracked_avoidance(self, position, velocity):
        """

        :param position:
        :param velocity:
        :return:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            if distance(position, neighbour.position) < self.collision_avoidance_d:
                part_vector = self.repulse_from_position(
                    position, neighbour.position, self.collision_avoidance_d)
                part_vector = unit(-velocity) * mag(part_vector)
                if self.debug:
                    self.logger.debug(
                        "Avoiding {0!s}:{1:f}:{2!s}".format(
                            neighbour.name, distance(position, neighbour.position), sixvec(part_vector)))
                force_vector += part_vector

                # Return an inverse vector to the obstacles
        if self.debug and not mag(force_vector) > 0.0:
            self.logger.debug("Repulse:{0!s}".format(force_vector))
        return self.normalize_behaviour(force_vector) * self.repulsive_factor

    @staticmethod
    def per_node_lawnmower(environment, swath, base_axis=0, altitude=None):
        """
        Generates a flat segmented lawnmower waypoint loop
        :param environment:
        :param swath:
        :param base_axis:
        :param altitude:
        """
        # Shape is 2D for waypoint generation

        try:
            front = max(environment[bool(base_axis)])
            back = min(environment[bool(base_axis)])
            left = min(environment[not bool(base_axis)])
            right = max(environment[not bool(base_axis)])
            if altitude is None and len(environment) != 3:
                raise ConfigError(
                    "altitude makes no sense for environment {0!s}".format(environment))
            elif altitude is None:
                altitude = np.diff(environment[-1])

            inc = np.sign(front - back)
            swath = inc * swath
        except Exception:
            logging.error("Error on per node lawnmower {0!s}".format(str(environment)))
            raise

        step = 0  # on a plateau going left or right (Odd-Rightward (max))
        stepping = 0  # on a rise going up or down (Odd-Upward (max))
        over_ratio = 8
        current_y = back - swath
        current_x = left - swath

        start = [current_x, current_y, altitude]

        if not stepping % 2:
            current_x = left - swath / over_ratio

        points = []

        while current_y < front + (stepping % 2) * swath:
            # four phases to a lawnmower iteration from back-left last point
            # 1) right to edge + swath
            # 2) up to step (if not above front + swath) else origin
            # 3) left to edge + swath
            # 4) up to step (if not above front + swath) else origin
            if stepping % 2:
                if step % 2:  # If on rightward leg
                    current_x = right + swath / over_ratio
                else:
                    current_x = left - swath / over_ratio
            else:
                # Intermediate apex to smooth turns
                if step % 2:  # If on rightward leg
                    points.append(
                        [current_x + (2 * swath), current_y + (swath / 2), altitude])
                else:
                    points.append(
                        [current_x - (2 * swath), current_y + (swath / 2), altitude])
                current_y += swath
                step += 1
            points.append([current_x, current_y, altitude])
            stepping += 1
        if bool(base_axis):
            points = np.asarray([[y, x, z] for (x, y, z) in points])
        else:
            points = np.asarray(points)

        return points

    def lawnmower(self, n, overlap=0, base_axis=0, twister=False, swath=100, prox=25):
        """
        N is either a single number (i.e. n rows of a shape) or a tuple (x, 1/y rows)
        :param n:
        :param overlap:
        :param base_axis:
        :param twister:
        :param swath:
        :param prox:
        """
        extent = np.asarray(zip(np.zeros(3), np.asarray(self.env_shape)))
        environment = np.asarray([self.env_shape[0:2] * (((vertex - 0.5) / 3) + 0.5)
                                  for vertex in np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])])

        front = np.max(environment, axis=0)[bool(base_axis)]
        back = np.min(environment, axis=0)[bool(base_axis)]
        right = np.max(environment, axis=0)[not bool(base_axis)]
        left = np.min(environment, axis=0)[not bool(base_axis)]
        top = max(extent[2])
        bottom = min(extent[2])
        mid_z = 2 * (top + bottom) / 3.0

        self.logger.info("Survey area:{0}km^2 ({1}), swath:{2}, overlap:{3}".format(
            (front - back) * (right - left) / 1e6,
            [front, back, right, left],
            swath, overlap)
        )

        inc = np.sign(front - back)
        swath = inc * swath

        if isinstance(n, tuple):
            row_count = n[base_axis]
            col_count = n[not base_axis]
        elif not np.sqrt(n) % 1 > 0:
            row_count = col_count = int(np.sqrt(n))
        elif n % 4 == 0:
            row_count = n / 4
            col_count = 4
        elif n % 3 == 0:
            row_count = n / 3
            col_count = 3
        elif n % 2 == 0:
            row_count = n / 2
            col_count = 2
        else:
            row_count = n
            col_count = 1

        row_height = (front - back) / col_count
        row_width = (right - left) / row_count

        courses = []

        for r, c in product(range(row_count), range(col_count)):
            sub_shape = [[(left + r * row_width) - overlap, (left + (r + 1) * row_width) + overlap],
                         [(back + c * row_height) - overlap, (back + (c + 1) * row_height) + overlap]]
            if twister:
                axis = base_axis + c % 2
            else:
                axis = base_axis
            courses.append(self.per_node_lawnmower(
                sub_shape, swath=swath, altitude=mid_z, base_axis=axis))

        self.waypoints = [waypoint(point, prox)
                          for point in courses[self.node.nodenum]]
        if not self.waypointloop:
            self.waypoints.append(
                waypoint(
                    agitate_position(
                        self.node.position.copy(), maximum=self.env_shape, var=prox * 2),
                    prox * 2)
            )
        self.nextwaypoint = 0


class FleetLawnmowerLoop(FleetLawnmower):
    """

    :param args:
    :param kwargs:
    """

    def __init__(self, *args, **kwargs):
        super(FleetLawnmowerLoop, self).__init__(*args, **kwargs)
        self.waypointloop = True


class Tail(Flock):
    """
    This behaviour gives the desire to be at the back of the fleet.
    This is accomplished by taking the incident angle between the clumping centre and the heading vector and taking the
        cross angle of it.
    This provides a 'braking' force along the axis of the fleets movement
    """

    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        self.behaviours.append(self.tail_vector)

    def tail_vector(self, position, velocity):
        """

        :param position:
        :param velocity:
        :return:
        """
        clumping_vector = self.clumping_vector(position, velocity)
        local_heading_vector = self.local_heading(position, velocity)
        force_vector = np.array([0, 0, 0], dtype=np.float)
        force_vector = -(clumping_vector + local_heading_vector)
        if self.debug:
            self.logger.debug("Tail:{0!s}".format(force_vector))
        return self.normalize_behaviour(force_vector) * self.clumping_factor


class SlowCoach(Flock):
    """
    This behaviour gives the desire to be clow.
    This is accomplished by the opposite of the last velocity
    This provides a 'braking' force along the axis of the nodes movement
    """

    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        self.behaviours.append(self.slowcoach_vector)

    def slowcoach_vector(self, position, velocity):
        """

        :param position:
        :param velocity:
        :return:
        """
        force_vector = np.array([0, 0, 0], dtype=np.float)
        force_vector = -velocity
        if self.debug:
            self.logger.debug("SlowCoach:{0!s}".format(force_vector))
        return self.normalize_behaviour(force_vector) * self.clumping_factor


####
# Malicious Class Aliases because I'm Lazy
####

# In the general case where everyone else is waypointing, the untargeted Flock behaviour is analogous
# to a un-initiated node 'shadowing' the fleet.
Shadow = Flock
