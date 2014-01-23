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
from itertools import product

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
        self.logger.info("creating Fleet instance with %d nodes" % len(nodes))
        Sim.Process.__init__(self, name="Fleet")
        self.nodes = nodes
        self.environment = simulation.environment
        self.simulation = simulation
        self.waiting = False

    def activate(self):
        for node in self.nodes:
            node.assignFleet(self)
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


class LawnmowerFleet(Fleet):
    def activate(self):
        extent = np.asarray([[0, 0], [1, 1]])
        prox = 10
        patrolcorners = [(self.environment.shape[0:2] * (((vertex - 0.5) / 3) + 0.5), prox) for vertex in extent]
        self.logger.error("Shape:{}".format(patrolcorners))

        courses = self.sharing_lawnmower(patrolcorners, len(self.nodes), overlap=10, twister=False)

        for node, course in zip(self.nodes, courses):
            node.activate(launch_args={'waypoints': course})
        Sim.activate(self, self.lifecycle())

    def sharing_lawnmower(self, shape, n, overlap=0, base_axis=0, twister=False):
        """
        N is either a single number (i.e. n rows of a shape) or a tuple (x, 1/y rows)
        """
        top = max(shape[base_axis])
        bottom = min(shape[base_axis])
        left = min(shape[not base_axis])
        right = max(shape[not base_axis])

        height = top - bottom
        width = right - left

        if isinstance(n, tuple):
            row_count = n[base_axis]
            col_count = n[not base_axis]
        else:
            row_count = n
            col_count = 1

        row_height = height / col_count
        row_width = width / row_count

        print("HW:{},{}".format(row_height, row_width))

        courses = []

        for r, c in product(range(row_count), range(col_count)):
            sub_shape = [[(left + r * row_width) - overlap, (left + (r + 1) * row_width) + overlap],
                         [(bottom + c * row_height) - overlap, (bottom + (c + 1) * row_height) + overlap]]
            print("rc:{},{},S:{}".format(r, c, sub_shape))
            if twister:
                axis = base_axis + c % 2
            else:
                axis = base_axis
            courses.append(self.lawnmower_waypoints(sub_shape, 5, base_axis=axis))
        return courses

    def lawnmower_waypoints(self, shape, swath, base_axis=0):
        """
        Generates an overlapping lawnmower grid across the environment
        """
        top = max(shape[base_axis])
        bottom = min(shape[base_axis])
        left = min(shape[not base_axis])
        right = max(shape[not base_axis])

        height = top - bottom
        inc = np.sign(height)
        swath = inc * swath

        step = 0 #on a plateau going left or right
        stepping = 0 # on a rise going up or down
        current_y = bottom
        current_x = left - swath

        start = [current_x, current_y]

        points = [start]

        while current_y < top + (1 + stepping % 2) * swath:
            # four phases to a lawnmower iteration from bottom-left last point
            # 1) right to edge + swath
            # 2) up to step (if not above top + swath) else origin
            # 3) left to edge + swath
            # 4) up to step (if not above top + swath) else origin
            if stepping % 2:
                if step % 2: # If on rightward leg
                    current_x = right + swath
                else:
                    current_x = left - swath
                points.append([current_x, current_y])
                stepping += 1
            else:
                current_y += swath
                points.append([current_x, current_y])
                step += 1
                stepping += 1
        if base_axis:
            points = np.asarray([[y, x] for (x, y) in points])
        else:
            points = np.asarray(points)
        return points

