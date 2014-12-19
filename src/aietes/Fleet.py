#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import sys
import traceback
from operator import attrgetter

import numpy as np
from scipy.spatial.distance import squareform, pdist

from aietes.Tools import Sim, distance, mag, seconds_to_str
from aietes.Tools.ProgressBar import ProgressBar
from aietes.Environment import Environment


try:
    from contrib.Ghia.uuv_time_delay_model import time_of_flight_matrix_complex

    ghia = True
except ImportError:
    ghia = False


# Local Debug
DEBUG = False


class Fleet(Sim.Process):
    """
    Fleets act initially as traffic managers for Nodes
    """

    def __init__(self, nodes, simulation, *args, **kwargs):
        self.logger = kwargs.get(
            "logger", simulation.logger.getChild(__name__))
        self.logger.info("creating Fleet instance with %d nodes" % len(nodes))
        Sim.Process.__init__(self, name="Fleet")
        self.nodes = nodes
        self.environment = simulation.environment
        self.shared_map = Environment(simulation,
                                      shape=self.environment.shape,
                                      base_depth=self.environment.depth,
                                      name="Shared")
        self.simulation = simulation
        self.waiting = False

    def activate(self):
        for node in self.nodes:
            node.assign_fleet(self)
        for node in self.nodes:
            node.activate()
        Sim.activate(self, self.lifecycle())

    def lifecycle(self):
        def all_nodes_passive():
            return all([n.passive() for n in self.nodes])

        def not_waiting():
            if self.simulation.waits:
                return not self.simulation.waiting
            else:
                return True

        if self.simulation.progress_display:
            try:
                from random import choice

                colors = ["BLUE", "GREEN", "CYAN", "RED", "MAGENTA", "YELLOW"]
                progress_bar = ProgressBar(
                    choice(colors), width=20, block='▣', empty='□')
            except TypeError as exp:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.logger.critical(
                    "Tried to start progress bar but failed with %s" % traceback.format_exc())
                progress_bar = None
            except:
                raise
        else:
            progress_bar = None

        # Canary for mission completeness
        uss_abraham_lincoln = True

        while True:
            self.simulation.waiting = True
            if DEBUG:
                self.logger.debug(
                    "Yield for all_nodes_passive: Environment Processing")
            yield Sim.waituntil, self, all_nodes_passive

            if DEBUG:
                self.logger.debug("Yield for all_nodes_passive: Node Processing")
            yield Sim.waituntil, self, all_nodes_passive

            # If all nodes have completed their missions, notify the user
            if self.is_mission_complete():
                if uss_abraham_lincoln:
                    self.logger.critical(
                        "Mission accomplished at {}".format(seconds_to_str(Sim.now())))
                    uss_abraham_lincoln = False
                    Sim.stopSimulation()

            # Pretty Printing
            if self.simulation.progress_display:
                percent_now = (
                    (100 * Sim.now()) / self.simulation.duration_intervals)
                if DEBUG and percent_now % 5 == 0:
                    self.logger.info("Fleet  %d%%: %s" %
                                     (percent_now, self.current_stats()))
                if not DEBUG and percent_now % 1 == 0 and progress_bar is not None:
                    progress_bar.render(int(percent_now),
                                        'step %s Processing %s...' % (percent_now, self.simulation.title))

            # Yield for anything the simulation should wait on (i.e. GUI)
            if DEBUG:
                self.logger.debug("Yield for not_waiting")
            yield Sim.waituntil, self, not_waiting

            # Perform any out of loop preprocessing required
            for node in self.nodes:
                Sim.reactivate(node)
            if DEBUG:
                self.logger.debug("Yield for all_nodes_passive: Fleet Updates")
            yield Sim.waituntil, self, all_nodes_passive

    def nodenum(self, node):
        """
        Return the index of the requested node
        """
        return node in self.nodes and self.nodes.index(node)

    def nodenum_from_id(self, node_id):
        """
        Return the index of the requested node node_id
        """
        return map(attrgetter('id'), self.nodes).index(node_id)

    def node_count(self):
        """
        Return the number of nodes in the fleet
        """
        return len(self.nodes)

    def node_names(self):
        """
        Return the node names in this fleet
        :return:
        """
        return [node.name for node in self.nodes]

    def node_positions(self, shared=True):
        """
        Return the fleet-list array of latest reported positions
        :param shared:
        (If shared: Use the 'drifted' reported positions)
        """
        if shared:
            latest_logs = self.shared_map.latest_logs()
        else:
            latest_logs = self.environment.latest_logs()
        positions = [None for _ in range(self.node_count())]
        times = [-1 for _ in range(self.node_count())]
        for id, log in latest_logs.items():
            index = self.nodenum_from_id(id)
            positions[index] = log.position
            times[index] = log.time
            if DEBUG:
                self.logger.debug("Node last seen at {} at {} @ {}".format(
                    log.name, log.position, log.time
                ))

        if len(set(times)) > 1:
            raise ValueError(
                "Latest shared logs not coalesced:{}".format(times))

        return np.asarray(positions)

    def node_positions_at(self, t, shared=True):
        """
        Return the fleet-list array of reported positions at a given time
        :param t:
        :param shared:
        """
        if shared:
            kb = self.shared_map.logs_at_time(t)
        else:
            kb = self.environment.logs_at_time(t)
        positions = [None for _ in range(self.node_count())]
        for id, log in kb.items():
            positions[self.nodenum_from_id(id)] = log.position
        return np.asarray(positions)

    def node_poslogs(self, shared=True):
        """
        Return the fleet-list array of reported position logs
        :param shared:
        """
        if shared:
            kb = self.shared_map
        else:
            kb = self.environment
        positions = [None for _ in range(self.node_count())]
        for nodeid in map(attrgetter('id'), self.nodes):
            positions[self.nodenum_from_id(nodeid)] = kb.node_pos_log(nodeid)
        return np.asarray(positions).swapaxes(2, 1)

    def node_cheat_drift_positions(self):
        """
        I hate this so much
        """
        return np.asarray([node.get_pos() for node in self.nodes])

    def node_cheat_positions(self):
        """
        I Hate this so much
        """
        return np.asarray([node.get_pos(true=True) for node in self.nodes])

    def node_cheat_last_ecea_estimates(self, update_index):
        """
        I Hate this so much
        :param update_index:
        """
        return np.asarray([node.ecea.pos_log[:, update_index] for node in self.nodes])

    def node_cheat_drift_positions_at(self, t):
        """
        I hate this so much
        :param t:
        """
        return np.asarray([node.pos_log[:, t] for node in self.nodes])

    def node_cheat_positions_at(self, t):
        """
        I Hate this so much
        :param t:
        """
        return np.asarray([node.drift.pos_log[:, t] for node in self.nodes])

    # noinspection PyNoneFunctionAssignment
    def node_position_errors(self, shared=True, error=0.001):
        """
        Fleet order Node position errors based on generic accuracy from origin of each node.

        THIS IS PERFECT IN THE Z-AXIS, DON'T USE FOR ANYTHING IMPORTANT
        :param shared:
        :param error:
        """
        original_positions = self.node_positions_at(0, shared=False)
        t = Sim.now()
        if t > 0:
            current_positions = self.node_positions(shared=shared)
        else:
            current_positions = original_positions.copy()

        delta = ((current_positions - original_positions) * error) + error
        # THIS IS A TERRIBLE HACK TO AVOID NANS IN THE WEIGHTING
        delta *= [1, 1, 0]
        return np.abs(delta)

    def time_of_flight_matrix(self, shared=False, speed_of_sound=1490.0, guess_index=0):
        """
        Fleet order time of flight matrix
        There are very few reasons why this would ever need to be from the shared database....
        (0-n,0-n)
        :param shared:
        :param speed_of_sound:
        :param guess_index:
        """
        latest_positions = self.node_cheat_positions()
        if guess_index > 0:
            if not ghia:
                raise NotImplementedError("This functionality requires the Ghia module")
            tof = time_of_flight_matrix_complex(
                latest_positions, self.environment.shape[2], guess_index)
        else:
            tof = squareform(pdist(latest_positions))
            tof /= speed_of_sound
        return tof

    def is_mission_complete(self):
        """
        Mission is complete when all waypoints finished and returned to mothership
        """
        on_mission = [n.on_mission for n in self.nodes]
        return not any(on_mission)

    def current_stats(self):
        """
        Print Current Vector Statistics
        """
        avgheading = np.array([0, 0, 0], dtype=np.float)
        fleetcenter = np.array([0, 0, 0], dtype=np.float)
        for node in self.nodes:
            avgheading += node.velocity
            fleetcenter += node.position

        avgheading /= float(len(self.nodes))
        fleetcenter /= float(len(self.nodes))

        maxdistance = np.float(0.0)
        maxdeviation = np.float(0.0)
        for node in self.nodes:
            maxdistance = max(
                maxdistance, distance(node.position, fleetcenter))
            v = node.velocity
            try:
                c = np.dot(avgheading, v) / mag(avgheading) / mag(v)
                maxdeviation = max(maxdeviation, np.arccos(c))
            except FloatingPointError:
                # In the event of v=0 (i.e. first time), fire back a - maxD
                # array.
                maxdeviation = avgheading

        return "V:%s,C:%s,D:%s,A:%s" % (avgheading, fleetcenter, maxdistance, maxdeviation)
