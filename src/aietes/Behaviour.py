#!/usr/bin/env python
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
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

from operator import attrgetter
import logging

import numpy as np

from aietes.Tools import map_entry, distance, fudge_normal, debug, unit, mag, listfix, sixvec, spherical_distance, ConfigError

#debug=True
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
        self.debug = debug
        self._start_log(self.node)
        if self.debug:
            self.logger.debug('from bev_config: %s' % self.bev_config)
            # self.logger.setLevel(logging.DEBUG)
        self.update_rate = 1
        self.memory = {}
        self.behaviours = []
        self.simulation = self.node.simulation
        self.env_shape = np.asarray(self.simulation.environment.shape)
        self.neighbours = {}
        self.positional_accuracy = listfix(float, self.bev_config.positional_accuracy)
        self.horizon = 200

    def _start_log(self, parent):
        self.logger = parent.logger.getChild("Bev:%s" % self.__class__.__name__)
        self.logger.debug('creating instance')

    def normalize_behaviour(self, forceVector):
        return forceVector

    def neighbour_map(self):
        """
        Returns a filtered map pulled from the map excluding self
        """
        orig_map = dict((k, v) for k, v in self.map.items() if v.object_id != self.node.id)

        # Internal Map of the neighbourhood based on best-guess of location (circa 5m / 5ms)
        for k, v in orig_map.items():
            orig_map[k].position = fudge_normal(v.position, self.positional_accuracy)
            orig_map[k].velocity = fudge_normal(v.velocity, self.positional_accuracy)

        return orig_map

    def addMemory(self, object_id, position, velocity):
        """
        Called by node lifecycle to update the internal representation of the environment
        """
        # TODO expand this to do SLAM?
        self.memory += map_entry(object_id, position, velocity)

    def process(self):
        """
        Process current map and memory information and update velocities
        """
        self.debug = debug and self.node == self.node.fleet.nodes[0]
        self.neighbours = self.neighbour_map()
        self.nearest_neighbours = self.getNearestNeighbours(self.node.position,
                                                            n_neighbours=self.n_nearest_neighbours)
        contributions = {}

        for behaviour in self.behaviours:
            try:
                contributions[behaviour.__name__] = behaviour(self.node.position, self.node.velocity)
            except Exception as exp:
                self.logger.error("%s(%s,%s)" % (behaviour.__name__, self.node.position, self.node.velocity))
                raise

        forceVector = sum(contributions.values())

        # TODO Under Drift, it's probably better to do wall-detection twice: once on node and once on environment
        forceVector = self.avoidWall(self.node.getPos(), self.node.velocity, forceVector)
        if self.debug:
            self.logger.debug("Response:%s" % forceVector)
        if self.debug:
            total = sum(map(mag, contributions.values()))

            self.logger.debug("contributions: %s of %3f" % (
                [
                    "%s:%.f%%" % (func, 100 * mag(value) / total)
                    for func, value in contributions.iteritems()
                ], total)
            )
        if self.debug:
            total = sum(map(mag, contributions.values()))
            for func, value in contributions.iteritems():
                self.logger.debug("%s:%.2f:%s" % (func, 100 * mag(value) / total, sixvec(value)))
        forceVector = fudge_normal(forceVector, 0.012)  # Random factor
        self.node.push(forceVector, contributions=contributions)
        return

    def getNearestNeighbours(self, position, n_neighbours=None, distance=np.inf):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        """
        # Sort and filter Neighbours from self.map by distance
        neighbours_with_distance = [map_entry(key,
                                              value.position, value.velocity,
                                              name=self.simulation.reverse_node_lookup(key).name,
                                              distance=self.node.distance_to(value.position)

        ) for key, value in self.neighbours.items()]
        # self.logger.debug("Got Distances: %s"%neighbours_with_distance)
        nearest_neighbours = sorted(neighbours_with_distance, key=attrgetter('distance')
        )
        # Select N neighbours in order
        # self.logger.debug("Nearest Neighbours:%s"%nearest_neighbours)
        if n_neighbours is not None:
            nearest_neighbours = nearest_neighbours[:n_neighbours]
        return nearest_neighbours

    def repulseFromPosition(self, position, repulsive_position, d_limit=1):
        forceVector = np.array([0, 0, 0], dtype=np.float)
        distanceVal = distance(position, repulsive_position)
        forceVector = unit(position - repulsive_position) * d_limit / float(distanceVal)

        if distanceVal < 2:
            raise RuntimeError("Too close to %s (%s) moving at %s; I was at %s moving at %s" %
                               (self.getNearestNeighbours(repulsive_position)[0].name,
                                self.getNearestNeighbours(repulsive_position)[0].position,
                                sixvec(self.getNearestNeighbours(repulsive_position)[0].velocity),
                                self.node.position,
                                sixvec(self.node.velocity)
                               ))
        if self.debug:
            self.logger.debug(
                "Repulsion from %s: %s, at range of %s" % (forceVector, repulsive_position, distanceVal))
        return forceVector

    def attractToPosition(self, position, attractive_position, d_limit=1):
        forceVector = np.array([0, 0, 0], dtype=np.float)
        distanceVal = distance(position, attractive_position)
        forceVector = unit(attractive_position - position) * (distanceVal / d_limit)
        if self.debug:
            self.logger.debug(
                "Attraction to %s: %s, at range of %s" % (forceVector, attractive_position, distanceVal))
        return forceVector

    def avoidWall(self, position, velocity, forceVector):
        """
        Called by responseVector to avoid walls to a distance of half min distance
        """
        response = np.zeros(shape=forceVector.shape)
        min_dist = self.neighbour_min_rad * 2
        avoid = False
        avoiding_position = None
        if np.any((np.zeros(3) + min_dist) > position):
            if self.debug:
                self.logger.debug("Too Close to the Origin-surfaces: %s" % position)
            offending_dim = position.argmin()
            avoiding_position = position.copy()
            avoiding_position[offending_dim] = float(0.0)
            avoid = True
        elif np.any(position > (self.env_shape - min_dist)):
            if self.debug:
                self.logger.debug("Too Close to the Upper-surfaces: %s" % position)
            offending_dim = position.argmax()
            avoiding_position = position.copy()
            avoiding_position[offending_dim] = float(self.env_shape[offending_dim])
            avoid = True
        else:
            response = forceVector

        if avoid:
            # response = 0.5 * (position-avoiding_position)
            try:
                response = self.repulseFromPosition(position, avoiding_position, 1)
            except RuntimeError:
                raise RuntimeError(
                    "Crashed out of environment with given position:{}, wall position:{}".format(position,
                                                                                                 avoiding_position))
                # response = (avoiding_position-position)
            self.logger.error("Wall Avoidance:%s" % response)

        return response


class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        self.n_nearest_neighbours = listfix(int, self.bev_config.nearest_neighbours)
        self.neighbourhood_max_rad = listfix(int, self.bev_config.neighbourhood_max_rad)
        self.neighbour_min_rad = listfix(int, self.bev_config.neighbourhood_min_rad)
        self.clumping_factor = self.bev_config.clumping_factor
        self.repulsive_factor = listfix(float, self.bev_config.repulsive_factor)
        self.schooling_factor = listfix(float, self.bev_config.schooling_factor)
        self.collision_avoidance_d = listfix(float, self.bev_config.collision_avoidance_d)

        self.behaviours.append(self.clumpingVector)
        self.behaviours.append(self.repulsiveVector)
        self.behaviours.append(self.localHeading)

        assert self.n_nearest_neighbours > 0
        assert self.neighbourhood_max_rad > 0
        assert self.neighbour_min_rad > 0

    def clumpingVector(self, position, velocity):
        """
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector = np.array([0, 0, 0], dtype=np.float)
        if len(self.nearest_neighbours) < 1:
            raise RuntimeError("I don't have any neighbours!")

        for neighbour in self.nearest_neighbours:
            vector += np.array(neighbour.position)

        try:
            # This assumes that the map contains one entry for each non-self node
            self.neighbourhood_com = vector / len(self.nearest_neighbours)
            if self.debug:
                self.logger.debug("Cluster Centre,position,factor,neighbours: %s,%s,%s,%s" % (
                    self.neighbourhood_com, vector, self.clumping_factor, len(self.nearest_neighbours)))
                # Return the fudged, relative vector to the centre of the cluster
            forceVector = self.attractToPosition(position, self.neighbourhood_com, self.collision_avoidance_d)
        except ZeroDivisionError:
            self.logger.error("Zero Division Error: Returning zero vector")
            forceVector = position - position
        except FloatingPointError:
            self.logger.error("FPE: vector=%s" % str(vector))
            raise

        if self.debug:
            self.logger.debug("Clump:%s" % forceVector)
        return self.normalize_behaviour(forceVector) * self.clumping_factor

    def repulsiveVector(self, position, velocity):
        """
        Represents the Short Range Repulsion behaviour:
            Steer away from it based on a repulsive desire curve
        """
        forceVector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            if distance(position, neighbour.position) < self.collision_avoidance_d:
                partVector = self.repulseFromPosition(position, neighbour.position, self.collision_avoidance_d)
                if self.debug:
                    self.logger.debug(
                        "Avoiding %s:%f:%s" % (
                            neighbour.name, distance(position, neighbour.position), sixvec(partVector)))
                forceVector += partVector

                # Return an inverse vector to the obstacles
        if self.debug:
            self.logger.debug("Repulse:%s" % forceVector)
        return self.normalize_behaviour(forceVector) * self.repulsive_factor

    def localHeading(self, position, velocity):
        """
        Represents Local Average Heading
        """
        vector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            vector += fudge_normal(unit(neighbour.velocity), max(abs(unit(neighbour.velocity))) / 3)
        forceVector = (vector / (len(self.nearest_neighbours)))
        if self.debug:
            self.logger.debug("Schooling:%s" % forceVector)
        if self.debug:
            self.logger.debug("V:%s,F:%s, %f" % (
                unit(velocity), unit(forceVector), spherical_distance(unit(velocity), unit(forceVector))))
        d = spherical_distance(unit(velocity), unit(forceVector))
        return self.normalize_behaviour(forceVector) * self.schooling_factor

    def _percieved_vector(self, node_id, time):
        """
        Finite Difference Estimation
        from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
        """
        node_history = sorted(filter(lambda x: x.object_id == node_id, self.memory), key=time)
        return (node_history[-1].position - node_history[-2].position) / (node_history[-1].time - node_history[-2].time)


class AlternativeFlockMixin():
    def __init__(self, *args, **kwargs):
        self.behaviours.remove(self.clumpingVector)
        self.behaviours.remove(self.repulsiveVector)
        self.behaviours.append(self.potentialVector)

    def potentialVector(self, position, velocity):
        """
        Computed singular collision/formation control vector instead of joint clumping/repulsive vectors
        """
        forceVector = np.array([0, 0, 0], dtype=np.float)
        for neighbour in self.nearest_neighbours:
            d = distance(position, neighbour.position)
            v = unit(neighbour.position - position)  # Direction to go
            if d < self.collision_avoidance_d:
                v = -v
            f = np.log(d) + (self.collision_avoidance_d / float(d))  # force to go
            nforceVector = f * v
            if self.debug:
                self.logger.debug("PotentialV: %s, for D:%s, F:%s, V:%s" % (nforceVector, d, f, v))
            forceVector += nforceVector
        return forceVector * self.clumping_factor


class AlternativeFlock(Flock, AlternativeFlockMixin):
    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        AlternativeFlockMixin.__init__(self, *args, **kwargs)


class WaypointMixin():
    """
    Waypoint MixIn Class defines the general waypoint behaviour and includes the inner 'waypoint' object class.
    """

    class waypoint(object):
        def __init__(self, positions, *args, **kwargs):
            """
            Defines waypoint paths:
                positions = [ [ position, proximity ], *]
            """
            self.logger = kwargs.get("logger", logging.getLogger(__name__))
            self.next = None
            (self.position, self.prox) = positions[0]
            if __debug__:
                self.logger.debug("Waypoint: %s,%s" % (self.position, self.prox))
            if len(positions) == 1:
                if __debug__:
                    self.logger.debug("End of Position List")
            else:
                self.append(positions[1:])

        def append(self, position):
            if self.next is None:
                self.next = WaypointMixin.waypoint(position)
            else:
                self.next.append(position)

        def insert(self, position):
            temp_waypoint = self.next
            self.next = WaypointMixin.waypoint(position)
            self.next.next = temp_waypoint

        def makeLoop(self, head):
            if self.next is None:
                self.next = head
            else:
                self.next.makeLoop(head)

    def __init__(self, *args, **kwargs):
        self.waypoint_factor = listfix(float, self.bev_config.waypoint_factor)
        self.waypoints = []
        self.nextwaypoint = None
        self.behaviours.append(self.waypointVector)

    def activate(self, *args, **kwargs):
        if not hasattr(self, str(self.bev_config.waypoint_style)):
            raise ConfigError("Cannot generate using waypoint definition:%s" % self.bev_config.waypoint_style)
        else:
            generator = attrgetter(str(self.bev_config.waypoint_style))
            g = generator(self)
            self.logger.debug("Generating waypoints: %s" % g.__name__)
            g()

    def patrolCube(self):
        """
        Generates a cubic patrol loop within the environment
        """
        shape = np.asarray(self.env_shape)
        prox = 50
        cubedef = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        )
        cubepatrolroute = [(shape * (((vertex - 0.5) / 3) + 0.5), prox) for vertex in cubedef]
        self.nextwaypoint = self.waypoint(cubepatrolroute)
        self.nextwaypoint.makeLoop(self.nextwaypoint)

        self.waypoints = cubepatrolroute

    def waypointVector(self, position, velocity):
        forceVector = np.array([0, 0, 0], dtype=np.float)
        if self.nextwaypoint is not None:
            neighbourhood_avg = sum(n.position for n in self.nearest_neighbours) / len(self.nearest_neighbours)
            real_d = distance(position, self.nextwaypoint.position)
            neighbourhood_d = distance(neighbourhood_avg, self.nextwaypoint.position)
            if neighbourhood_d < self.nextwaypoint.prox:
                # GRANT ACHIEVEMENT
                self.node.grantAchievement((self.nextwaypoint.position, real_d))
                if __debug__:
                    self.logger.info("Moving to Next waypoint:%s" % self.nextwaypoint.position)
                self.nextwaypoint = self.nextwaypoint.next
            forceVector = self.attractToPosition(position, self.nextwaypoint.position, self.nextwaypoint.prox)
        return self.normalize_behaviour(forceVector) * self.waypoint_factor


class Waypoint(Flock, WaypointMixin):
    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        WaypointMixin.__init__(self, *args, **kwargs)


class AlternativeWaypoint(AlternativeFlock, Waypoint):
    def __init__(self, *args, **kwargs):
        AlternativeFlock.__init__(self, *args, **kwargs)
        WaypointMixin.__init__(self, *args, **kwargs)


class FleetWaypointer(Flock, WaypointMixin):
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

    #TODO The guts of this are in notebook

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        self.n_nearest_neighbours = listfix(int, self.bev_config.nearest_neighbours)
        self.neighbourhood_max_rad = listfix(int, self.bev_config.neighbourhood_max_rad)
        self.neighbour_min_rad = listfix(int, self.bev_config.neighbourhood_min_rad)
        self.repulsive_factor = listfix(float, self.bev_config.repulsive_factor)
        self.collision_avoidance_d = listfix(float, self.bev_config.collision_avoidance_d)

        self.behaviours.append(self.repulsiveVector)

        assert self.n_nearest_neighbours > 0
        assert self.neighbourhood_max_rad > 0
        assert self.neighbour_min_rad > 0

    def activate(self, *args, **kwargs):
        course = kwargs.get('waypoints', None)
        if course is None:
            raise ConfigError("No waypoints given to Waypoint Follower")

        self.nextwaypoint = self.waypoint(course)
        self.nextwaypoint.makeLoop(self.nextwaypoint)

        self.waypoints = course


class Tail(Flock):
    """
    This behaviour gives the desire to be at the back of the fleet.
    This is accomplished by taking the incident angle between the clumping centre and the heading vector and taking the
        cross angle of it.
    This provides a 'braking' force along the axis of the fleets movement
    """

    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        self.behaviours.append(self.tailVector)

    def tailVector(self, position, velocity):
        clumpingVector = self.clumpingVector(position, velocity)
        localheadingVector = self.localHeading(position, velocity)
        forceVector = np.array([0, 0, 0], dtype=np.float)
        forceVector = -(clumpingVector + localheadingVector)
        if self.debug:
            self.logger.debug("Tail:%s" % forceVector)
        return self.normalize_behaviour(forceVector) * self.clumping_factor


class SlowCoach(Flock):
    """
    This behaviour gives the desire to be clow.
    This is accomplished by the opposite of the last velocity
    This provides a 'braking' force along the axis of the nodes movement
    """

    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        self.behaviours.append(self.slowcoachVector)

    def slowcoachVector(self, position, velocity):
        forceVector = np.array([0, 0, 0], dtype=np.float)
        forceVector = -(velocity)
        if self.debug:
            self.logger.debug("SlowCoach:%s" % forceVector)
        return self.normalize_behaviour(forceVector) * self.clumping_factor

####
# Malicious Class Aliases because I'm Lazy
####

#In the general case where everyone else is waypointing, the untargeted Flock behaviour is analogous
#   to a un-initiated node 'shadowing' the fleet.
Shadow = Flock
