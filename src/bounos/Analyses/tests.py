# coding=utf-8
from unittest import TestCase

__author__ = 'bolster'

import numpy as np
from pandas.util.testing import assert_frame_equal
from time import time
import pandas as pd

import aietes
import aietes.Tools as Tools
import bounos.Analyses.Trust as Trust


class TestGenerateNodeTrustPerspective(TestCase):
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


class TestGenerateTrustLogsFromCommsLogs(TestCase):
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


class TestTrustGenerationEquality(TestCase):
    @classmethod
    def setUpClass(cls):
        control_case_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Control-2015-07-31-07-56-18"
        with pd.get_store(control_case_path + '.h5') as store:
            tf = store.trust
        cls.sample = tf.xs('CombinedTrust', level='var', drop_level=False).xs(0, level='run', drop_level=False).xs(
            'Bravo', level='observer', drop_level=False)

    def testUnweighted(self):
        sample_weight = None

        tp_as_matrix = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=False,
                                                             as_matrix=True)
        tp = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=False,
                                                   as_matrix=False)
        assert_frame_equal(tp, tp_as_matrix)
        tpp_as_matrix = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=True,
                                                              as_matrix=True)
        assert_frame_equal(tpp_as_matrix, tp_as_matrix)
        tpp = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=True,
                                                    as_matrix=False)
        assert_frame_equal(tpp, tpp_as_matrix)
        assert_frame_equal(tp, tpp)
        print(self.sample.shape)

    def testWeighted(self):
        sample_weight = pd.Series(np.random.randn(len(self.sample.keys())), index=self.sample.keys())

        tp_as_matrix = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=False,
                                                             as_matrix=True)
        tp = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=False,
                                                   as_matrix=False)
        assert_frame_equal(tp, tp_as_matrix)
        tpp_as_matrix = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=True,
                                                              as_matrix=True)
        assert_frame_equal(tpp_as_matrix, tp_as_matrix)
        tpp = Trust.generate_node_trust_perspective(self.sample, metric_weights=sample_weight, par=True,
                                                    as_matrix=False)
        assert_frame_equal(tpp, tpp_as_matrix)
        assert_frame_equal(tp, tpp)
        print(self.sample.shape)
