import unittest
import polybos


class PolybosScenarioGeneration(unittest.TestCase):
    def RatioExperimentGeneration(self):
        """
        Basic tests of polybos experiment generation
        """
        count = 4
        behaviour = "Flocking"
        e = polybos.ExperimentManager(count=count)
        e.addRatioScenario(behaviour)
        self.assertEqual(len(e.scenarios), count + 1)
        s = e.scenarios[count / 2]
        self.assertEqual(len(s.get_behaviour_dict()[behaviour]), count / 2)


if __name__ == "__main__":
    unittest.main()
