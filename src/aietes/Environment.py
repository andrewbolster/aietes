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
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

from collections import namedtuple
from operator import attrgetter

import numpy as np
from SimPy import Simulation as Sim

from aietes.Tools import MapEntry, distance, DEBUG, ConfigError

Log = namedtuple('Log', ['name', 'object_id', 'time', 'position'])


class Environment(object):
    """
    Environment Class representing the physical environment inc any objects
    / activities within that environment that are not controlled by the
    simulated entities i.e. wind, tides, speed of sound at depth, etc
    """

    def __init__(self, simulation, shape=None, resolution=1, base_depth=-1000, sos_model=None, name="", **kwargs):
        """
        Generate a box with points from 0 to (size) in each dimension, where
        each point represents a cube of side resolution metres:
                Volume is the representation of the physical environment (XYZ)
                Map is the
        """
        self._start_log(simulation, name)
        self.map = {}
        self.pos_log = []
        self.depth = base_depth
        self.sos = 1400
        self.simulation = simulation
        if shape is None or not isinstance(shape, np.ndarray):
            try:
                shape = np.asarray(shape)
                if shape is None or not isinstance(shape, np.ndarray) or len(shape) != 3:
                    raise ConfigError(
                        "Shape doesn't make sense: {0}{1}".format(shape, type(shape)))
            except:
                raise
        self.shape = shape

    def _start_log(self, parent, name):
        self.logger = parent.logger.getChild(
            "{0}{1}".format(name, self.__class__.__name__))
        self.logger.debug('creating instance')

    def random_position(self, want_empty=True, on_a_plane=False, buff=30):
        """
        Return a random empty map reference within the environment volume
        :param want_empty:
        :param on_a_plane:
        :param buff:
        """
        is_empty = False
        if on_a_plane:
            z_plane = self.shape[2] / 2.0

        while not is_empty:
            ran_x = np.random.uniform(buff, self.shape[0] - buff)
            ran_y = np.random.uniform(buff, self.shape[1] - buff)
            ran_z = np.random.uniform(
                buff, self.shape[2] - buff) if not on_a_plane else z_plane
            candidate_pos = self.position_around(
                np.asarray((ran_x, ran_y, ran_z)), on_a_plane=on_a_plane)
            is_empty = self.is_empty(candidate_pos)

        return candidate_pos

    def position_around(self, position=None, stddev=30, on_a_plane=False):
        """
        Return a nearly-random map entry within the environment volume around a given position
        :param position:
        :param stddev:
        :param on_a_plane:
        """
        if position is None:
            position = np.asarray(self.shape) / 2
        if isinstance(position, basestring):
            if position == "surface":
                position = np.zeros(3)
                position[0] = 3 * stddev
                position[1] = 3 * stddev
                position[2] = self.shape[2] - (2 * stddev)
            else:
                raise ValueError("Incorrect position string")

        candidate_pos = None

        if self.is_outside(position):
            raise ValueError(
                "Position is not within volume: {0}".format(position))
        else:
            valid = False
            while not valid:
                candidate_pos = np.random.normal(
                    np.asarray(position), [stddev, stddev, stddev / 3])
                candidate_pos = np.asarray(candidate_pos, dtype=int)
                # if generating a position on a plane, retain the zaxis
                if on_a_plane:
                    candidate_pos[2] = position[2]
                valid = self.is_safe(candidate_pos, 50)
                if DEBUG:
                    self.logger.debug(
                        "Candidate position: {0!s}:{1!s}".format(candidate_pos, valid))
        return tuple(candidate_pos)

    def is_outside(self, position, tolerance=10):
        too_high = any(position > self.shape - tolerance)
        too_low = any(position < tolerance)
        return too_high or too_low

    def is_safe(self, position, tolerance=30):
        return self.is_empty(position, tolerance=tolerance) and not self.is_outside(position, tolerance=tolerance)

    def is_empty(self, position, tolerance=10):
        """
        Return if a given position is 'empty' for a given proximity tolerance
        :param position:
        :param tolerance:
        """
        distances = [
            distance(position, entry.position) > tolerance for entry in self.map]
        return all(distances)

    def update(self, object_id, position, velocity):
        """
        Update the environment to reflect a movement
        :param object_id:
        :param position:
        :param velocity:
        """
        object_name = self.simulation.reverse_node_lookup(object_id).name
        t = Sim.now()

        if np.isnan(np.sum(position)):
            raise RuntimeError("Invalid Position Update from {name}@{time}:{pos}".format(name=object_name,
                                                                                         time=t,
                                                                                         pos=position)
                               )

        # DEBUG=True
        if t < self.simulation.duration_intervals:
            try:
                assert self.map[
                           object_id].position is not position, "Attempted direct obj=obj comparison"
                update_distance = distance(
                    self.map[object_id].position, position)
                if DEBUG:
                    self.logger.debug("Moving {0!s} {1:f} from {2!s} to {3!s} @ {4:d}".format(object_name,
                                                                           update_distance,
                                                                           self.map[
                                                                               object_id].position,
                                                                           position, t))
                self.map[object_id] = MapEntry(
                    object_id, position, velocity, object_name)
            except KeyError:
                if DEBUG:
                    self.logger.debug(
                        "Creating map entry for {0!s} at {1!s} @ {2:d}".format(object_name, position, Sim.now()))
                self.map[object_id] = MapEntry(
                    object_id, position, velocity, object_name)
            self.pos_log.append(Log(name=object_name,
                                    position=position,
                                    object_id=object_id,
                                    time=t
                                    ))
        else:
            self.logger.debug(
                "Reaching end of simulation: Dropping {0}th frame for array size consistency (0->{1}={2})".format(t, t,
                                                                                                               t + 1))

    def node_pos_log(self, uid):
        """
        Returns the poslog (3,t) for a node of a given uuid
        :param uid:
        """
        pos_get = attrgetter('position')
        node_get = lambda l: l.object_id == uid
        pos_log = np.asarray(map(pos_get, filter(node_get, self.pos_log)))
        return pos_log

    def logs_at_time(self, t):
        """
        Return the object logs for the fleet at a given time
        :param t:
        """
        return {l.object_id: l for l in filter(lambda l: l.time == t, self.pos_log)}

    def latest_logs(self):
        """
        Returns the latest positions (n,3) for the nodes in the fleet
        #todo this shouldn't be so stupid; keep a 'latest'
        """
        last_log = {oid: None for oid in self.object_ids()}
        for log in reversed(self.pos_log):
            if last_log[log.object_id] is None:
                last_log[log.object_id] = log
                if DEBUG:
                    self.logger.debug(
                        "Object {0} last seen at {1} @ {2}".format(log.name, log.position, log.time))
            if None not in last_log.values():
                break

        return last_log

    def object_ids(self):
        """
        Returns the number of unique objects in the fleet's shared map
        """
        return set(map(attrgetter('object_id'), self.pos_log))

    def generate_point_plane(self, index=-1):
        """
        Calculate the current best fit plane between all nodes
        :param index:
        """
        pos_get = attrgetter('position')
        positions = map(pos_get, self.map.values())
        n = len(positions)
        average = sum(positions) / n
        covariant = np.cov(np.asarray(positions - average), rowvar=0)
        evecs, evals = np.linalg.eig(covariant)
        sorted_evecs = evecs[evals.argsort()]
        return average, sorted_evecs[index]

    @staticmethod
    def generate_normal_plane(point, normal):  # plot final plane
        """

        :param point:
        :param normal:
        :return:
        """
        d = np.dot(-point, normal)
        [xx, yy] = np.meshgrid(np.arange(point[0] - 10, point[0] + 10),
                               np.arange(point[1] - 10, point[1] + 10))
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        return xx, yy, zz

    def eigen_plot(self, index=-1):
        """

        :param index:
        :return:
        """
        average, normal = self.generate_point_plane(index)
        return self.generate_normal_plane(average, normal)

    def export(self, filename=None):
        """
        Export the current environment to a csv
        :param filename:
        """
        assert filename is not None
        np.savez(filename, self.pos_log)
