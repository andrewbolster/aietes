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

"""Unit tests for Polybos"""

import unittest
import polybos


class ScenarioGeneration(unittest.TestCase):
    def testRatioExperimentGeneration(self):
        """Basic tests of polybos experiment generation"""
        count = 4
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count)
        e.addRatioScenario(behaviour)
        self.assertEqual(len(e.scenarios), count + 1)
        s = e.scenarios[count / 2]
        self.assertEqual(len(s.getBehaviourDict()[behaviour]), count / 2)

    def testSimulationStatsQuery(self):
        """ Test that generateSimulationStats does single and multiple query responses based on data"""
        count = 4
        runcount = 2
        runtime = 100
        behaviour = "Flock"
        e = polybos.ExperimentManager(node_count=count)
        e.addRatioScenario(behaviour)
        e.run(runcount=runcount,
              runtime=runtime)
        multistats = e.generateSimulationStats()
        self.assertEqual(len(multistats), len(e.scenarios))
        for stats in multistats:
            self.assertEqual(len(stats), runcount)


if __name__ == "__main__":
    unittest.main()
