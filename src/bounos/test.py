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

"""Unit tests for bounos"""
import logging
import unittest
import tempfile
import os.path

import numpy as np

import aietes
import bounos
from aietes.Tools import _results_dir as default_results_dir


class LookupMethods(unittest.TestCase):
    def setUp(self):
        """
        Populate the generated results dir with a basic simulation
        :return:
        """
        self.run_time = 100
        self.simulation = aietes.Simluation(logtoconsole=logging.ERROR, progress_display=False)
        self.prep_dict = self.simulation.prepare(sim_time=self.run_time)
        self.sim_time = self.simulation.simulate()

    def testDirSearchOrder(self):
        """
        Ensure that npz_in_dir returns in order of modification
        :return:
        """
        file_list = bounos.npz_in_dir(default_results_dir)
        mtime_list = map(os.path.getmtime, file_list)
        self.assertTrue(all(x <= y for x, y in zip(mtime_list, mtime_list[1:])), mtime_list)

    def testDirSearchValid(self):
        """
        Ensure that npz_in_dir returns only npz files
        :return:
        """
        file_list = bounos.npz_in_dir(default_results_dir)
        self.assertTrue(all(x.endswith('.npz') for x in file_list))

    def testDirSearchValidOnEmpty(self):
        empty_dir = tempfile.mkdtemp()
        sources = bounos.npz_in_dir(empty_dir)
        self.assertIsInstance(sources, list, 'Sources should be list type:{}'.format(type(sources)))
        self.assertItemsEqual(sources, [], 'Sources should be empty for empty dir: {}'.format(sources))

    def testDirSearchInvalidOnNonExist(self):
        fake_dir = tempfile.mkdtemp()
        os.rmdir(fake_dir)
        self.assertRaises(OSError, bounos.npz_in_dir, fake_dir)


class DataPackageCreation(unittest.TestCase):
    def testSourceParsing(self):
        """ Ensure that DataPackage is correctly created by using a sourcefile: 
            does NOT test numerical validitity of anything
        """
        # Use tempfile to make up an arbitrary name, which is then automatically
        # suffixed with .npz
        source_file_obj = tempfile.NamedTemporaryFile(delete=False)
        source_file = source_file_obj.name
        source_file_obj.close()
        print source_file
        s_n_nodes = 10
        s_tmax = 1000
        positions = []
        vectors = []
        names = []
        for n in range(s_n_nodes):
            positions.append(np.empty((3, s_tmax)))
            vectors.append(np.empty((3, s_tmax)))
            names.append("testname")

        contributions = [
                            [None for _ in range(s_tmax)] for _ in range(s_n_nodes)],
        achievements = [[] for _ in range(s_n_nodes)]
        environment = np.empty(3, dtype=int),
        config = {'testconfig': True},
        title = "TESTCONFIG"

        np.savez(source_file,
                 p=positions,
                 v=vectors,
                 names=names,
                 contributions=contributions,
                 achievements=achievements,
                 environment=environment,
                 config=config,
                 title=title
        )
        dp = bounos.DataPackage(source=source_file + '.npz')
        print dp.p.shape
        self.assertEqual(dp.title, title)
        self.assertEqual(dp.n, s_n_nodes)
        self.assertEqual(dp.tmax, s_tmax)
