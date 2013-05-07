""" Unit test for aietes """

import unittest

import logging
import aietes
import bounos

datapackage_per_node_members = ['p', 'v', 'names', 'contributions', 'achievements']


class DefaultBehaviour(unittest.TestCase):
    def setUp(self):
        """Aietes should simulate fine with no input by pulling in from default values"""
        self.simulation = aietes.Simulation(logtoconsole=logging.ERROR, progress_display=False)
        self.prep_dict = self.simulation.prepare()
        self.sim_time = self.simulation.simulate()

    def testDictAndTimeReporting(self):
        """Simulation time at prep should be same as after simulation"""
        self.assertEqual(self.prep_dict['sim_time'], self.sim_time)

    def testDataPackage(self):
        """Test generation of DataPackage and the availability of data members"""
        datapackage = self.simulation.generateDataPackage()
        self.assertIsInstance(datapackage, bounos.DataPackage)
        for member in datapackage_per_node_members:
            m_val = getattr(datapackage, member)
            self.assertIsNotNone(m_val, msg=member)
            self.assertEqual(len(m_val), datapackage.n)


if __name__ == "__main__":
    unittest.main() 
