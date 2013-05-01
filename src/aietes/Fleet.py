#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np

from aietes.Tools import Sim, distance, mag
from aietes.Tools.ProgressBar import ProgressBar


class Fleet(Sim.Process):

    """
    Fleets act initially as traffic managers for Nodes
    """

    def __init__(self, nodes, simulation, *args, **kwargs):
        self.logger = kwargs.get("logger", simulation.logger.getChild(__name__))
        self.logger.info("creating instance")
        Sim.Process.__init__(self, name="Fleet")
        self.nodes = nodes
        self.environment = simulation.environment
        self.simulation = simulation
        self.waiting = False

    def activate(self):
        Sim.activate(self, self.lifecycle())
        for node in self.nodes:
            node.activate()

    def lifecycle(self):
        def allPassive():
            return all([n.passive() for n in self.nodes])

        def not_waiting():
            if self.simulation.waits:
                return not self.simulation.waiting
            else:
                return True

        progress_bar = ProgressBar('green', width=20, block='▣', empty='□')
        self.logger.info("Initialised Node Lifecycle")
        while True:
            self.simulation.waiting = True
            yield Sim.waituntil, self, allPassive
            if self.simulation.progress_display:
                percent_now = ((100 * Sim.now()) / self.simulation.duration_intervals)
                if __debug__ and percent_now % 5 == 0:
                    self.logger.info("Fleet  %d%%: %s" % (percent_now, self.currentStats()))
                if not __debug__ and percent_now % 1 == 0:
                    progress_bar.render(int(percent_now),
                                        'step %s\nProcessing %s...' % (percent_now, self.simulation.title))
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
