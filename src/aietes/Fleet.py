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

from aietes.Tools import Sim, distance, mag
from aietes.Tools.ProgressBar import ProgressBar

#Local Debug
debug = False

class Fleet(Sim.Process):
    """
    Fleets act initially as traffic managers for Nodes
    """

    def __init__(self, nodes, simulation, *args, **kwargs):
        self.logger = kwargs.get("logger", simulation.logger.getChild(__name__))
        self.logger.info("creating Fleet instance with %d nodes"%len(nodes))
        Sim.Process.__init__(self, name="Fleet")
        self.nodes = nodes
        self.environment = simulation.environment
        self.simulation = simulation
        self.waiting = False

    def activate(self):
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
                progress_bar = ProgressBar('green', width=20, block='▣', empty='□')
            except TypeError as exp:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.logger.info("Tried to start progress bar but failed with %s" % traceback.format_exc())
                progress_bar = None
        else:
            progress_bar = None
        self.logger.info("Initialised Fleet Lifecycle")
        while True:
            self.simulation.waiting = True
            if debug: self.logger.debug("Yield for allPassive")
            yield Sim.waituntil, self, allPassive
            if self.simulation.progress_display:
                percent_now = ((100 * Sim.now()) / self.simulation.duration_intervals)
                if __debug__ and percent_now % 5 == 0:
                    self.logger.info("Fleet  %d%%: %s" % (percent_now, self.currentStats()))
                if not __debug__ and percent_now % 1 == 0 and progress_bar is not None:
                    progress_bar.render(int(percent_now),
                                        'step %s\nProcessing %s...' % (percent_now, self.simulation.title))
            if debug: self.logger.debug("Yield for not_waiting")
            yield Sim.waituntil, self, not_waiting
            for node in self.nodes:
                Sim.reactivate(node)

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
