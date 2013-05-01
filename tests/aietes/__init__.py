""" Unit test for aietes """

import aietes
import unittest

class DefaultBehaviour(unittest.TestCase):
    def testDefaultSimluationExecution(self):
        """Aietes should simulate fine with no input by pulling in from default values"""
        s = aietes.Simulation()
        prep_dict = s.prepare()
        self.assertIsInstance(prep_dict, dict)
        sim_time = s.simulate()
        self.assertEqual(prep_dict['sim_time'], sim_time)
if __name__ == "__main__":
    unittest.main() 
