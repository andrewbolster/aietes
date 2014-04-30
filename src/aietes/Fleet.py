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
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import sys
import traceback

import numpy as np
from scipy.spatial.distance import squareform, pdist

from operator import attrgetter

from aietes.Tools import Sim, distance, mag, secondsToStr
from aietes.Tools.ProgressBar import ProgressBar
from aietes.Environment import Environment



#Local Debug
debug = False


class Fleet(Sim.Process):
    """
    Fleets act initially as traffic managers for Nodes
    """

    def __init__(self, nodes, simulation, *args, **kwargs):
        self.logger = kwargs.get("logger", simulation.logger.getChild(__name__))
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
            node.assignFleet(self)
        for node in self.nodes:
            node.activate()
        Sim.activate(self, self.lifecycle())

    def lifecycle(self):
        def allPassive():
            return all([n.passive() for n in self.nodes])

        def not_waiting():
            if self.simulation.waits:
                return not self.simulation.waiting
            else:
                return True

        if self.simulation.progress_display:
            try:
                from random import choice
                colors = ["BLUE","GREEN","CYAN","RED","MAGENTA","YELLOW"]
                progress_bar = ProgressBar(choice(colors), width=20, block='▣', empty='□')
            except TypeError as exp:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.logger.critical("Tried to start progress bar but failed with %s" % traceback.format_exc())
                progress_bar = None
            except:
                raise
        else:
            progress_bar = None

        # Canary for mission completeness
        USS_Abraham_Lincoln=True

        while True:
            self.simulation.waiting = True
            if debug: self.logger.debug("Yield for allPassive: Node Processing")
            yield Sim.waituntil, self, allPassive

            # If all nodes have completed their missions, notify the user
            if self.isMissionComplete():
                if USS_Abraham_Lincoln:
                    self.logger.critical("Mission accomplished at {}".format(secondsToStr(Sim.now())))
                    USS_Abraham_Lincoln=False
                    Sim.stopSimulation()

            # Pretty Printing
            if self.simulation.progress_display:
                percent_now = ((100 * Sim.now()) / self.simulation.duration_intervals)
                if debug and percent_now % 5 == 0:
                    self.logger.info("Fleet  %d%%: %s" % (percent_now, self.currentStats()))
                if not debug and percent_now % 1 == 0 and progress_bar is not None:
                    progress_bar.render(int(percent_now),
                                        'step %s Processing %s...' % (percent_now, self.simulation.title))

            # Yield for anything the simulation should wait on (i.e. GUI)
            if debug: self.logger.debug("Yield for not_waiting")
            yield Sim.waituntil, self, not_waiting

            # Perform any out of loop preprocessing required
            for node in self.nodes:
                Sim.reactivate(node)
            if debug: self.logger.debug("Yield for allPassive: Process Updates")
            yield Sim.waituntil, self, allPassive

    def nodenum(self,node):
        """
        Return the index of the requested node
        """
        return node in self.nodes and self.nodes.index(node)

    def nodenum_from_id(self, id):
        """
        Return the index of the requested node id
        """
        return map(attrgetter('id'), self.nodes).index(id)


    def nodeCount(self):
        """
        Return the number of nodes in the fleet
        """
        return len(self.nodes)

    def nodePositions(self, shared=True):
        """
        Return the fleet-list array of latest reported positions
        (If shared: Use the 'drifted' reported positions)
        """
        if shared:
            latest_logs = self.shared_map.latest_logs()
        else:
            latest_logs = self.environment.latest_logs()
        positions = [ None for _ in range(self.nodeCount())]
        times = [ -1 for _ in range(self.nodeCount())]
        for id,log in latest_logs.items():
            index = self.nodenum_from_id(id)
            positions[index]=log.position
            times[index]=log.time
            if debug: self.logger.debug("Node last seen at {} at {} @ {}".format(
                log.name, log.position, log.time
            ))

        if len(set(times))>1:
            raise ValueError("Latest shared logs not coalesced:{}".format(times))

        return np.asarray(positions)

    def nodePositionsAt(self, t, shared=True):
        """
        Return the fleet-list array of reported positions at a given time
        """
        if shared:
            kb = self.shared_map.logs_at_time(t)
        else:
            kb = self.environment.logs_at_time(t)
        positions = [ None for _ in range(self.nodeCount())]
        for id, log in kb.items():
            positions[self.nodenum_from_id(id)] = log.position
        return np.asarray(positions)


    def nodePosLogs(self, shared=True):
        """
        Return the fleet-list array of reported position logs
        """
        if shared:
            kb = self.shared_map
        else:
            kb = self.environment
        positions = [ None for _ in range(self.nodeCount())]
        for nodeid in map(attrgetter('id'),self.nodes):
            positions[self.nodenum_from_id(nodeid)] = kb.node_pos_log(nodeid)
        return np.asarray(positions).swapaxes(2,1)

    def nodeCheatDriftPositions(self):
        """
        I hate this so much
        """
        return np.asarray([node.getPos() for node in self.nodes])

    def nodeCheatPositions(self):
        """
        I Hate this so much
        """
        return np.asarray([node.getPos(true=True) for node in self.nodes])

    def nodeCheatLastECEAEstimates(self, update_index):
        """
        I Hate this so much
        """
        return np.asarray([node.ecea.pos_log[:, update_index] for node in self.nodes])

    def nodeCheatDriftPositionsAt(self, t):
        """
        I hate this so much
        """
        return np.asarray([node.pos_log[:,t] for node in self.nodes])

    def nodeCheatPositionsAt(self, t):
        """
        I Hate this so much
        """
        return np.asarray([node.drift.pos_log[:,t] for node in self.nodes])

    def nodePositionErrors(self, shared=True, error=0.001):
        """
        Fleet order Node position errors based on generic accuracy from origin of each node.

        THIS IS PERFECT IN THE Z-AXIS, DON'T USE FOR ANYTHING IMPORTANT
        """
        original_positions = self.nodePositionsAt(0, shared=False)
        t = Sim.now()
        if t>0:
            current_positions = self.nodePositions(shared=shared)
        else:
            current_positions = original_positions.copy()

        delta = ((current_positions-original_positions)*error) + error
        delta *=[1,1,0] # THIS IS A TERRIBLE HACK TO AVOID NANS IN THE WEIGHTING
        return np.abs(delta)


    def timeOfFlightMatrix(self, shared=False, speed_of_sound=1490.0):
        """
        Fleet order time of flight matrix
        There are very few reasons why this would ever need to be from the shared database....
        (0-n,0-n)
        """
        latest_positions = self.nodeCheatPositions()
        tof = squareform(pdist(latest_positions))
        tof /= speed_of_sound
        return tof

    def isMissionComplete(self):
        """
        Mission is complete when all waypoints finished and returned to mothership
        """
        on_mission = [n.on_mission for n in self.nodes]
        return not any(on_mission)

    def currentStats(self):
        """
        Print Current Vector Statistics
        """
        avgHeading = np.array([0, 0, 0], dtype=np.float)
        fleetCenter = np.array([0, 0, 0], dtype=np.float)
        for node in self.nodes:
            avgHeading += node.velocity
            fleetCenter += node.position

        avgHeading /= float(len(self.nodes))
        fleetCenter /= float(len(self.nodes))

        maxDistance = np.float(0.0)
        maxDeviation = np.float(0.0)
        for node in self.nodes:
            maxDistance = max(maxDistance, distance(node.position, fleetCenter))
            v = node.velocity
            try:
                c = np.dot(avgHeading, v) / mag(avgHeading) / mag(v)
                maxDeviation = max(maxDeviation, np.arccos(c))
            except FloatingPointError:
                # In the event of v=0 (i.e. first time), fire back a - maxD array.
                maxDeviation = avgHeading

        return "V:%s,C:%s,D:%s,A:%s" % (avgHeading, fleetCenter, maxDistance, maxDeviation)


