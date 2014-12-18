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

"""Unit tests for ChartBuilders"""

import unittest

from matplotlib.figure import Figure

import aietes
import bounos
import bounos.ChartBuilders
from aietes.Tools import _results_dir as default_results_dir


class ChartBuilders(unittest.TestCase):
    def setUp(self):
        """
        Load the most recent results file
        :return:
        """
        # Make up a datapackage and hope for the best
        sim = aietes.Simulation(title=self.__class__.__name__,
                                config_file=aietes.Tools.getConfigFile('bella_static.conf'),
                                progress_display=False)
        sim.prepare(sim_time=1000)
        sim.simulate()
        self.dp=sim.generateDataPackage()


    def test_lost_packet_distribution(self):
        """
        Ensure lost_packet_distribution doesn't crash
        :return:
        """
        if not self.dp.has_comms_data():
            raise unittest.SkipTest("Latest DataPackage has no Comms data: {}".format(self.dp.title))

        test_figure = bounos.ChartBuilders.lost_packet_distribution(self.dp)
        self.assertIsInstance(test_figure, Figure)

    def test_end_to_end_delay_distribution(self):
        """
        Ensure end_to_end_delay_distribution doesn't crash
        :return:
        """
        if not self.dp.has_comms_data():
            raise unittest.SkipTest("Latest DataPackage has no Comms data: {}".format(self.dp.title))

        test_figure = bounos.ChartBuilders.end_to_end_delay_distribution(self.dp)
        self.assertIsInstance(test_figure, Figure)

    def test_source_and_dest_delay_violin_plot(self):
        """
        Ensure source_and_dest_delay_violin_plot doesn't crash
        :return:
        """
        if not self.dp.has_comms_data():
            raise unittest.SkipTest("Latest DataPackage has no Comms data: {}".format(self.dp.title))

        test_figure = bounos.ChartBuilders.source_and_dest_delay_violin_plot(self.dp)
        self.assertIsInstance(test_figure, Figure)

    def test_channel_occupancy_distribution(self):
        """
        Ensure channel_occupancy_distribution doesn't crash
        :return:
        """
        if not self.dp.has_comms_data():
            raise unittest.SkipTest("Latest DataPackage has no Comms data: {}".format(self.dp.title))

        test_figure = bounos.ChartBuilders.channel_occupancy_distribution(self.dp)
        self.assertIsInstance(test_figure, Figure)

    @unittest.skip('This Graph is Deprecated')
    def test_combined_trust_observation_summary(self):
        """
        Ensure combined_trust_observation_summary doesn't crash
        :return:
        """
        if not self.dp.has_trust_data():
            raise unittest.SkipTest("Latest DataPackage has no Trust data: {}".format(self.dp.title))

        test_figure = bounos.ChartBuilders.combined_trust_observation_summary(self.dp, target=self.dp.names[1])
        self.assertIsInstance(test_figure, Figure)

