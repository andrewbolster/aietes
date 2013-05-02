""" Unit test for aietes """

import unittest
import logging

import aietes
import bounos


class DefaultBehaviour(unittest.TestCase):
    def setUp(self):
        """Aietes should simulate fine with no input by pulling in from default values"""
        self.simulation = aietes.Simulation(logtoconsole=logging.ERROR, progress_display=False)
        self.prep_dict = self.simulation.prepare()
        self.sim_time = self.simulation.simulate()

    def testDictAndTimeReporting(self):
        self.assertEqual(self.prep_dict['sim_time'], self.sim_time)

    def testDataPackage(self):
        datapackage = self.simulation.generateDataPackage()
        self.assertIsInstance(datapackage, bounos.DataPackage)


if __name__ == "__main__":
    unittest.main() 
