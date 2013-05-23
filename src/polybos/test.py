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
