# coding=utf-8
from unittest import TestCase

__author__ = 'bolster'

import numpy as np
from pandas.util.testing import assert_frame_equal

import aietes
import aietes.Tools as Tools
import bounos.Analyses.Trust as Trust


class TestGenerate_node_trust_perspective(TestCase):
    def setUp(self):
        # Make up a datapackage and hope for the best
        sim = aietes.Simulation(title=self.__class__.__name__,
                                config_file=aietes.Tools.get_config_file('bella_static.conf'),
                                progress_display=False)
        sim.prepare(sim_time=1250)
        sim.simulate()
        dp = sim.generate_datapackage()
        self.trust = Trust.generate_trust_logs_from_comms_logs(dp.comms['logs'])


    def test_generate_node_trust_perspective_par_equiv(self):
        lin = Trust.generate_node_trust_perspective(self.trust, par=False)
        par = Trust.generate_node_trust_perspective(self.trust, par=True)
        assert_frame_equal(par, lin)

    def test_perspective_shapes(self):
        tp = Trust.generate_node_trust_perspective(self.trust)
        self.assertSequenceEqual(tp.index.names, ['var', 'run', 'observer', 't'],
                                 "Should have 'var run observer t' index")
        self.assertTrue(all(tp.columns == ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']),
                        "Columns should contain node names")
        self.assertEqual(tp.columns.name, "target",
                         "Column should be called target")


class TestGenerate_trust_logs_from_comms_logs(TestCase):
    def setUp(self):
        # Make up a datapackage and hope for the best
        sim = aietes.Simulation(title=self.__class__.__name__,
                                config_file=aietes.Tools.get_config_file('bella_static.conf'),
                                progress_display=False)
        sim.prepare(sim_time=1250)
        sim.simulate()
        self.dp = sim.generate_datapackage()

    def test_generate_trust_logs_from_comms_logs(self):
        tf = Trust.generate_trust_logs_from_comms_logs(self.dp.comms['logs'])
        self.assertEqual(len(tf.index.levels), 3, "Should have three levels")

        self.assertSequenceEqual(tf.index.names, ['observer', 'target', 't'],
                                 "Called observer target t in that order")
        self.assertTrue(np.all(tf.groupby(level='target').size() == 15),
                        "Should have 6 targets total with 15 observations each (i.e. 3*metric)")
