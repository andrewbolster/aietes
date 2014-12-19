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

"""Unit tests for Polybos"""

import unittest
import re

import polybos


class ExperimentGeneration(unittest.TestCase):
    def testRatioExperimentGeneration(self):
        """Basic tests of polybos experiment generation"""
        count = 4
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count)
        e.add_ratio_scenarios(behaviour)
        self.assertEqual(len(e.scenarios), count + 1)
        v, s = e.scenarios.items()[count / 2]
        self.assertEqual(len(s.get_behaviour_dict()[behaviour]), int(count * float(re.split('\(|\)|\%', v)[1]) / 100.0))

    def testRuntimeModification(self):
        """Ensure that polybos appropriatly propogates simulation time using the run method"""
        count = 4
        runcount = 1
        runtime = 10
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count)
        e.add_ratio_scenarios(behaviour)
        e.run(runcount=runcount,
              runtime=runtime,
              retain_data=True)
        for k, v in e.scenarios.items():
            self.assertEqual(v.datarun[0].tmax, runtime,
                             "Simulation time ({}) should match requested time ({})".format(
                                 v.datarun[0].tmax, runtime))

    def testRuntimeModificationParallel(self):
        """Ensure that polybos appropriatly propogates simulation time using the run method"""
        count = 4
        runcount = 1
        runtime = 10
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count, parallel=True)
        e.add_ratio_scenarios(behaviour)
        e.run(runcount=runcount,
              runtime=runtime,
              retain_data=True)
        for k, v in e.scenarios.items():
            self.assertEqual(v.datarun[0].tmax, runtime,
                             "Simulation time ({}) should match requested time ({})".format(
                                 v.datarun[0].tmax, runtime))

    # @unittest.skipIf(os.name == 'nt', "Skipping MultiCore test as it appears to be broken under Windows...")
    def testParSim(self):
        """ Test Multiprocessing Execution """
        count = 4
        runcount = 4
        runtime = 10
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count, parallel=True)
        e.add_ratio_scenarios(behaviour)
        e.run(runcount=runcount,
              runtime=runtime)
        multistats = e.generate_simulation_stats()
        self.assertEqual(len(e.scenarios), count + 1,
                         "Scenarios count for behaviour ratio should be node_count ({}) + 1, is {}".format(
                             count, len(e.scenarios)))
        self.assertIsInstance(multistats, dict,
                              "MultiStats should be a Dict, is type {}".format(
                                  type(multistats)))
        self.assertEqual(len(multistats), len(e.scenarios),
                         "Multistats({}) should be same length as scenarios({})".format(
                             len(multistats), len(e.scenarios)))
        for runname, stats in multistats.iteritems():
            self.assertEqual(len(stats), runcount,
                             "N of stats for {} ({}) should match runcount ({})".format(
                                 runname, len(stats), runcount))

    def testSimulationStatsQuery(self):
        """ Test that generate_simulation_stats does single and multiple query responses based on data"""
        count = 4
        runcount = 2
        runtime = 10
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count)
        e.add_ratio_scenarios(behaviour)
        e.run(runcount=runcount,
              runtime=runtime)
        multistats = e.generate_simulation_stats()
        self.assertEqual(len(e.scenarios), count + 1)
        self.assertEqual(len(multistats), len(e.scenarios))
        for stats in multistats.items():
            self.assertEqual(len(stats), runcount)


if __name__ == "__main__":
    unittest.main()
