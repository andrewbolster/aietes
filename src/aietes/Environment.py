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
from collections import namedtuple
import numpy as np

from SimPy import Simulation as Sim

from aietes.Tools import map_entry, distance, debug, ConfigError


Log = namedtuple('Log', ['name', 'object_id', 'time', 'position'])


class Environment():
    """
    Environment Class representing the physical environment inc any objects
    / activities within that environment that are not controlled by the
    simulated entities i.e. wind, tides, speed of sound at depth, etc
    """

    def __init__(self, simulation, shape=None, resolution=1, base_depth=-1000, sos_model=None, **kwargs):
        """
        Generate a box with points from 0 to (size) in each dimension, where
        each point represents a cube of side resolution metres:
                Volume is the representation of the physical environment (XYZ)
                Map is the
        """
        self._start_log(simulation)
        self.map = {}
        self.pos_log = []
        self.depth = base_depth
        self.sos = 1400
        self.simulation = simulation
        if shape is None or not isinstance(shape, np.ndarray):
            try:
                shape = np.asarray(shape)
                if shape is None or not isinstance(shape, np.ndarray) or len(shape) != 3:
                    raise ConfigError("Shape doesn't make sense: {}{}".format(shape, type(shape)))
            except:
                raise
        self.shape = shape

    # TODO Random Surface Generation
    # self.generateSurface()
    # TODO 'Tidal motion' factor

    def _start_log(self, parent):
        self.logger = parent.logger.getChild("%s" % self.__class__.__name__)
        self.logger.debug('creating instance')

    def is_empty(self, position, tolerance=1):
        """
        Return if a given position is 'empty' for a given proximity tolerance
        """
        distances = [distance(position, entry.position) > tolerance for entry in self.map]
        return all(distances)

    def random_position(self, want_empty=True):
        """
        Return a random empty map reference within the environment volume
        """
        is_empty = False
        while not is_empty:
            ran_x = np.random.uniform(0, self.shape[0])
            ran_y = np.random.uniform(0, self.shape[1])
            ran_z = np.random.uniform(0, self.shape[2])
            candidate_pos = (ran_x, ran_y, ran_z)
            is_empty = self.is_empty(candidate_pos)

        return candidate_pos

    def position_around(self, position=None, stddev=30):
        """
        Return a nearly-random map entry within the environment volume around a given position
        """
        if position is None:
            position = np.asarray(self.shape) / 2
        if isinstance(position,basestring):
            if position == "surface":
                position = np.asarray(self.shape)/2
                position[2]=self.shape[2] - (3 * stddev)
            else:
                raise ValueError("Incorrect position string")

        candidate_pos = None

        if self.is_outside(position):
            raise ValueError("Position is not within volume: {}".format(position))
        else:
            valid = False
            while not valid:
                candidate_pos = np.random.normal(np.asarray(position), stddev)
                candidate_pos = np.asarray(candidate_pos, dtype=int)
                try:
                    valid = self.is_empty(candidate_pos) and not self.is_outside(candidate_pos)
                except:
                    raise TypeError("{}".format(candidate_pos))
                if debug:
                    self.logger.debug("Candidate position: %s:%s" % (candidate_pos, valid))
        return tuple(candidate_pos)

    def is_outside(self, position):
        too_high = not all(position < self.shape)
        too_low = not all(position > 0)
        return too_high or too_low

    def update(self, object_id, position, velocity):
        """
        Update the environment to reflect a movement
        """
        object_name = self.simulation.reverse_node_lookup(object_id).name
        try:
            assert self.map[object_id].position is not position, "Attempted direct obj=obj comparison"
            update_distance = distance(self.map[object_id].position, position)
            if __debug__ and debug:
                self.logger.debug("Moving %s %f from %s to %s" % (object_name,
                                                                  update_distance,
                                                                  self.map[object_id].position,
                                                                  position))
            self.map[object_id] = map_entry(object_id, position, velocity, object_name)
        except KeyError:
            if __debug__ and debug:
                self.logger.debug("Creating map entry for %s at %s" % (object_name, position))
            self.map[object_id] = map_entry(object_id, position, velocity, object_name)
        self.pos_log.append(Log(name=object_name,
                                position=position,
                                object_id=object_id,
                                time=Sim.now()
        ))

    def pointPlane(self, index=-1):
        """
        Calculate the current best fit plane between all nodes
        """
        pos_get = attrgetter('position')
        positions = map(pos_get, self.map.values())
        N = len(positions)
        average = sum(positions) / N
        covariant = np.cov(np.asarray(positions - average), rowvar=0)
        evecs, evals = np.linalg.eig(covariant)
        sorted_evecs = evecs[evals.argsort()]
        return average, sorted_evecs[index]

    def normalPlane(self, point, normal):  # plot final plane
        d = np.dot(-point, normal)
        [xx, yy] = np.meshgrid(np.arange(point[0] - 10, point[0] + 10),
                               np.arange(point[1] - 10, point[1] + 10))
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        return xx, yy, zz

    def eigenPlot(self, index=-1):
        average, normal = self.pointPlane(index)
        return self.normalPlane(average, normal)

    def export(self, filename=None):
        """
        Export the current environment to a csv
        """
        assert filename is not None
        np.savez(filename, self.pos_log)
