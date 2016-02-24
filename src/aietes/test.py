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

""" Unit test for aietes """
import logging
import os
import shutil
import unittest
import tempfile
from pprint import pformat

import matplotlib

matplotlib.use("Agg")

import aietes
import bounos

datapackage_per_node_members = [
    'p', 'v', 'names', 'contributions', 'achievements']


class DefaultBehaviour(unittest.TestCase):
    def setUp(self):
        """Aietes should simulate fine with no input by pulling in from default values"""
        self.run_time = 100
        self.title = self.__class__.__name__
        try:
            self.simulation = aietes.Simulation(
                title=self.title,
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
        datapackage = self.simulation.generate_datapackage()
        self.assertIsInstance(datapackage, bounos.DataPackage)
        for member in datapackage_per_node_members:
            m_val = getattr(datapackage, member)
            self.assertIsNotNone(m_val, msg=member)
            self.assertEqual(len(m_val), datapackage.n)

    def testDataPackageDestination(self):
        """Test generation of DataPackage files and ensure they go in the right place!"""
        empty_dir = tempfile.mkdtemp()
        output_dict = self.simulation.postprocess(output_file=self.__class__.__name__,
                                                  output_path=empty_dir,
                                                  data_file=True)
        expected_filename = "{0}.aietes".format(self.__class__.__name__)
        expected_path = os.path.join(empty_dir, expected_filename)
        self.assertEqual(output_dict['data_file'], expected_path + '.npz',
                         "DataFile Paths don't match {0}:{1}".format(output_dict['data_file'], expected_path + '.npz'))
        self.assertEqual(output_dict['config_file'], expected_path + '.conf',
                         "ConfigFile Paths don't match {0}:{1}".format(output_dict['config_file'],
                                                                     expected_path + '.npz'))
        self.assertTrue(os.path.isfile(expected_path + '.npz'),
                        "Didn't store datapackage in generated temp directory {0}".format(expected_path))
        self.assertTrue(os.path.isfile(expected_path + '.conf'),
                        "Didn't store conf file in generated temp directory {0}".format(expected_path))
        shutil.rmtree(empty_dir)

    def testDefaultDataPackageDestination(self):
        """Test generation of DataPackage files and ensure they go in the right place!"""
        output_dict = self.simulation.postprocess(output_file=self.__class__.__name__,
                                                  data_file=True)
        expected_filename = "{0}.aietes".format(self.__class__.__name__)
        expected_path = os.path.join(aietes.Tools._results_dir, expected_filename)
        self.assertEqual(output_dict['data_file'], os.path.abspath(expected_path + '.npz'),
                         "DataFile Paths don't match {0}:{1}".format(output_dict['data_file'], expected_path + '.npz'))
        self.assertEqual(output_dict['config_file'], os.path.abspath(expected_path + '.conf'),
                         "ConfigFile Paths don't match {0}:{1}".format(output_dict['config_file'],
                                                                     expected_path + '.npz'))
        self.assertTrue(os.path.isfile(expected_path + '.npz'),
                        "Didn't store datapackage in generated temp directory {0}".format(expected_path))
        self.assertTrue(os.path.isfile(expected_path + '.conf'),
                        "Didn't store conf file in generated temp directory {0}".format(expected_path))
        os.remove(expected_path + '.npz')
        os.remove(expected_path + '.conf')


class ConfigBehaviour(unittest.TestCase):
    @unittest.skip("Reminder for later")
    def testZeroFleetCreation(self):
        """Ensure failure on launching fleet with 0 nodes"""
        # TODO
        pass


@unittest.skip("Broken but should be revisited: Subprocess bug in matplotlib")
class OutputBehaviour(unittest.TestCase):
    def testGifGeneration(self):
        """Ensure nothing goes too wrong with gif generation"""
        options = aietes.option_parser().defaults
        options.update({'gif': True, 'quiet': False, 'sim_time': 100, 'output_path': tempfile.mkdtemp()})
        options = aietes.Tools.Dotdictify(options)
        try:
            print(pformat(options))
            aietes.go(options)
        except Exception as e:
            raise

    def testMovieGeneration(self):
        """Ensure nothing goes too wrong with movie generation"""
        options = aietes.option_parser().defaults
        options.update({'movie': True, 'quiet': False, 'sim_time': 100, 'output_path': tempfile.mkdtemp()})
        options = aietes.Tools.Dotdictify(options)
        print(pformat(options))
        aietes.go(options)


class Tools(unittest.TestCase):
    """Slow grow of tool testing"""

    def testGetConfigFile_Empty(self):
        self.assertRaises(TypeError, aietes.Tools.get_config_file)

    def testGetConfigFile_NonExist(self):
        self.assertRaises(OSError, aietes.Tools.get_config_file, "NotThere.conf")

    def testGetConfigFile_Exists(self):
        config_tail = 'bella_static.conf'
        config_path = aietes.Tools.get_config_file(config_tail)
        config_list = map(os.path.abspath,
                          [os.path.join(aietes.Tools._config_dir, c)
                           for c in os.listdir(aietes.Tools._config_dir)
                           ]
                          )
        self.assertIn(config_path, config_list)
        self.assertTrue(config_path.endswith(config_tail))

    def testGetConfig_Default(self):
        config = aietes.Tools.get_config()
        self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()
