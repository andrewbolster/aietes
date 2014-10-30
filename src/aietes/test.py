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
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

""" Unit test for aietes """
import logging
import unittest
from pprint import pformat

from aietes.Tools import dotdictify
import aietes
import bounos


datapackage_per_node_members = [
    'p', 'v', 'names', 'contributions', 'achievements']


class DefaultBehaviour(unittest.TestCase):

    def setUp(self):
        """Aietes should simulate fine with no input by pulling in from default values"""
        count = 4
        self.run_time = 100
        try:
            self.simulation = aietes.Simulation(
                logtoconsole=logging.ERROR, progress_display=False)
            self.prep_dict = self.simulation.prepare(sim_time=self.run_time)
            self.sim_time = self.simulation.simulate()
        except RuntimeError:
            print "Got Runtime Error on SetUp, trying one more time"
            self.simulation = aietes.Simulation(
                logtoconsole=logging.ERROR, progress_display=False)
            self.prep_dict = self.simulation.prepare(sim_time=self.run_time)
            self.sim_time = self.simulation.simulate()

    def testDictAndTimeReporting(self):
        """Simulation time at prep should be same as after simulation"""
        self.assertEqual(self.prep_dict['sim_time'], self.run_time)
        self.assertEqual(self.prep_dict['sim_time'], self.sim_time)

    def testDataPackage(self):
        """Test generation of DataPackage and the availability of data members"""
        datapackage = self.simulation.generateDataPackage()
        self.assertIsInstance(datapackage, bounos.DataPackage)
        for member in datapackage_per_node_members:
            m_val = getattr(datapackage, member)
            self.assertIsNotNone(m_val, msg=member)
            self.assertEqual(len(m_val), datapackage.n)


class ConfigBehaviour(unittest.TestCase):

    @unittest.skip("Reminder for later")
    def testZeroFleetCreation(self):
        """Ensure failure on launching fleet with 0 nodes"""
        # TODO
        pass


class OutputBehaviour(unittest.TestCase):

    def testGifGeneration(self):
        """Ensure nothing goes too wrong with gif generation"""
        options = aietes.option_parser().defaults
        options.update({'gif': True, 'quiet': False, 'sim_time': 100})
        options = dotdictify(options)
        try:
            print(pformat(options))
            aietes.go(options)
        except Exception as e:
            raise

    def testMovieGeneration(self):
        """Ensure nothing goes too wrong with movie generation"""
        options = aietes.option_parser().defaults
        options.update({'movie': True, 'quiet': False, 'sim_time': 100})
        options = dotdictify(options)
        print(pformat(options))
        aietes.go(options)


if __name__ == "__main__":
    unittest.main()
