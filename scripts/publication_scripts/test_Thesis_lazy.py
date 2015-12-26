# coding=utf-8
from __future__ import division

from scripts.publication_scripts import get_mobility_stats

__author__ = 'bolster'

# coding: utf-8

# # Required Modules and Configurations

# In[1]:

import os
from os.path import expanduser
import itertools
import functools
import unittest
import logging
import tempfile
import shutil

logging.basicConfig()

import pandas as pd
import numpy as np
import sklearn.ensemble as ske
import matplotlib.pylab as plt
from matplotlib import rc_context
import matplotlib.ticker as plticker

loc_25 = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals

import aietes
import aietes.Tools as Tools

from bounos.ChartBuilders import plot_nodes, latexify
import bounos.Analyses.Trust as Trust
from bounos.ChartBuilders import weight_comparisons, radar, format_axes

# pylab.rcParams['figure.figsize'] = 16, 12

##################
# USING TEMP DIR #
##################
use_temp_dir = False

texify_cols = 1
texify_factor = 1

selected_scenarios = [
#    'single_mobile',
#    'allbut1_mobile',
    'bella_all_mobile',
    'bella_static',
]

class ThesisLazyDiagrams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        if use_temp_dir:
            cls.dirpath = tempfile.mkdtemp()
        else:
            cls.dirpath = expanduser("~/src/thesis/Figures")

        os.chdir(cls.dirpath)
        # # Plot Style Config

        _boxplot_kwargs = {
            'showmeans': True,
            'showbox': False,
            'widths': 0.2,
            'linewidth': 2
        }

        cls.malicious = "MaliciousBadMouthingPowerControlTrustMedianTests-0.025-3-2015-02-19-23-27-01.h5"
        cls.good = "TrustMedianTests-0.025-3-2015-02-19-23-29-39.h5"
        cls.selfish = "MaliciousSelfishTargetSelectionTrustMedianTests-0.025-3-2015-03-29-19-32-36.h5"
        cls.outlier = "outliers.h5"
        cls.generated_files = []

    def setUp(self):

        for filename in [self.good, self.malicious, self.selfish]:
            if not os.path.isfile(Tools.in_results(filename)):
                self.fail("No file {}".format(filename))

    def tearDown(self):
        logging.info("Successfully Generated:\n{}".format(self.generated_files))
        if use_temp_dir:
            shutil.rmtree(self.dirpath, ignore_errors=True)

    def testPhysicalNodeLayout(self):
        # # Graph: Physical Layout of Nodes
        required_files = ["s1_layout.pdf"]
        #
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        latexify(columns=texify_cols, factor=texify_factor) 

        base_config = aietes.Simulation.populate_config(
            aietes.Tools.get_config('bella_static.conf'),
            retain_default=True
        )
        texify = lambda t: "${}_{}$".format(t[0], t[1])
        node_positions = {texify(k): np.asarray(v['initial_position'], dtype=float) for k, v in
                          base_config['Node']['Nodes'].items() if 'initial_position' in v}
        node_links = {0: [1, 2, 3], 1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 5], 3: [0, 1, 4], 4: [1, 3, 5], 5: [1, 2, 4]}

        fig = plot_nodes(node_positions, figsize=(4, 1.6), node_links=node_links, radius=3, scalefree=True,
                         square=False)
        fig.tight_layout(pad=0.3)
        fig.savefig("s1_layout.pdf", transparent=True)
        plt.close(fig)

        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)

    def testThroughputLines(self):
        # # Plot Throughput Lines
        required_files = [
            "throughput_sep_lines_static.pdf",
            "throughput_sep_lines_all_mobile.pdf"
        ]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        latexify(columns=texify_cols, factor=texify_factor) 

        for mobility in ['static', 'all_mobile']:
            df = get_mobility_stats(mobility)
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            for (k, g), ls in zip(df.groupby('separation'), itertools.cycle(["-", "--", "-.", ":"])):
                ax.plot(g.rate, g.throughput, label=k, linestyle=ls)
            ax.legend(loc="upper left")
            ax.set_xlabel("Packet Emission Rate (pps)")
            ax.set_ylabel("Avg. Throughput (bps)")
            fig.tight_layout()
            fig.savefig("throughput_sep_lines_{}.pdf".format(mobility), transparent=True, facecolor='white')
            plt.close(fig)

        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)

    def testMTFMBoxplots(self):
        # # MTFM Boxplots
        required_files = [
            "trust_bella_static_fair.pdf",
            "trust_bella_all_mobile_fair.pdf",
            "trust_bella_static_malicious.pdf",
            "trust_bella_all_mobile_malicious.pdf",
            "trust_bella_static_selfish.pdf",
            "trust_bella_all_mobile_selfish.pdf"
        ]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        figsize = latexify(columns=texify_cols, factor=texify_factor) 

        weight_comparisons.plot_mtfm_boxplot(self.good, keyword="fair",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True, prefix="")
        weight_comparisons.plot_mtfm_boxplot(self.malicious, keyword="malicious",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True, prefix="")
        weight_comparisons.plot_mtfm_boxplot(self.selfish, keyword="selfish",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True, prefix="")
        for f in required_files:
            self.assertTrue(os.path.isfile(f), f)
            self.generated_files.append(f)

    def testWeightComparisons(self):
        # # Weight Comparisons

        required_files = [
            "trust_bella_static_emph_ADelay_BadMouthingPowerControl.pdf",
            "trust_bella_static_emph_ATXP_BadMouthingPowerControl.pdf",
            "trust_bella_static_emph_RXThroughput_BadMouthingPowerControl.pdf",
            "trust_bella_static_emph_TXThroughput_BadMouthingPowerControl.pdf",
            "trust_bella_all_mobile_emph_ADelay_BadMouthingPowerControl.pdf",
            "trust_bella_all_mobile_emph_ARXP_BadMouthingPowerControl.pdf",
            "trust_bella_all_mobile_emph_ATXP_BadMouthingPowerControl.pdf",
            "trust_bella_all_mobile_emph_RXThroughput_BadMouthingPowerControl.pdf",
            "trust_bella_all_mobile_emph_TXThroughput_BadMouthingPowerControl.pdf",
            "trust_bella_static_emph_ADelay_SelfishTargetSelection.pdf",
            "trust_bella_static_emph_ATXP_SelfishTargetSelection.pdf",
            "trust_bella_static_emph_RXThroughput_SelfishTargetSelection.pdf",
            "trust_bella_static_emph_TXThroughput_SelfishTargetSelection.pdf",
            "trust_bella_all_mobile_emph_ADelay_SelfishTargetSelection.pdf",
            "trust_bella_all_mobile_emph_ARXP_SelfishTargetSelection.pdf",
            "trust_bella_all_mobile_emph_ATXP_SelfishTargetSelection.pdf",
            "trust_bella_all_mobile_emph_RXThroughput_SelfishTargetSelection.pdf",
            "trust_bella_all_mobile_emph_TXThroughput_SelfishTargetSelection.pdf"
        ]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        figsize = latexify(columns=texify_cols, factor=texify_factor) 

        for s in selected_scenarios:
            weight_comparisons.plot_weight_comparisons(self.good, self.malicious,
                                                       malicious_behaviour="BadMouthingPowerControl",
                                                       s=s, figsize=figsize, show_title=False,
                                                       labels=["Fair", "Malicious"],
                                                       prefix=""
                                                       )
            weight_comparisons.plot_weight_comparisons(self.good, self.selfish,
                                                       malicious_behaviour="SelfishTargetSelection",
                                                       s=s, figsize=figsize, show_title=False,
                                                       labels=["Fair", "Selfish"],
                                                       prefix=""
                                                       )
            weight_comparisons.plot_weight_comparisons(self.good, self.selfish,
                                                       malicious_behaviour="SelfishTargetSelection",
                                                       s=s, figsize=figsize, show_title=False,
                                                       labels=["Fair", "Selfish"],
                                                       prefix=""
                                                       )
        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)

    def testSummaryGraphsForMalGdScenarios(self):
        # # Summary Graphs with malicious, selfish and fair scenarios
        required_files = ["trust_beta_otmf_fair.pdf", "trust_beta_otmf_malicious.pdf", "trust_beta_otmf_selfish.pdf"]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        latexify(columns=texify_cols, factor=texify_factor) 

        def beta_trusts(trust, length=4096):
            #TODO This should be optimised to not use the same dataframe
            trust['+'] = (trust.TXThroughput / length) * (1 - trust.PLR)
            trust['-'] = (trust.TXThroughput / length) * trust.PLR
            beta_trust = trust[['+', '-']].unstack(level='target')
            trust.drop(['+','-'], axis=1, inplace=True)
            return beta_trust

        def beta_calcs(beta_trust):
            beta_trust = pd.stats.moments.ewma(beta_trust, span=2)
            beta_t_confidence = lambda s, f: 1 - np.sqrt((12 * s * f) / ((s + f + 1) * (s + f) ** 2))
            beta_t = lambda s, f: s / (s + f)
            otmf_t = lambda s, f: 1 - np.sqrt(
                ((((beta_t(s, f) - 1) ** 2) / 2) + (((beta_t_confidence(s, f) - 1) ** 2) / 9))) / np.sqrt(
                (1 / 2) + (1 / 9))
            beta_vals = beta_trust.apply(lambda r: beta_t(r['+'], r['-']), axis=1)
            otmf_vals = beta_trust.apply(lambda r: otmf_t(r['+'], r['-']), axis=1)
            return beta_vals, otmf_vals

        def plot_beta_mtmf_comparison(beta_trust, mtfm, key):

            beta, otmf = beta_calcs(beta_trust)
            fig, ax = plt.subplots(1, 1, sharey=True)
            ax.plot(beta.xs('n0', level='observer')['n1'].values, label="Hermes", linestyle='--')
            ax.plot(otmf.xs('n0', level='observer')['n1'].values, label="OTMF", linestyle=':')
            ax.plot(mtfm.values, label="MTFM")
            ax.set_ylim((0, 1))
            ax.set_ylabel("Trust Value".format(key))
            ax.set_xlabel("Observation")
            ax = format_axes(ax)
            ax.legend(loc='lower center', ncol=3)
            ax.axhline(mtfm.mean(), color="r", linestyle='-.')
            ax.yaxis.set_major_locator(loc_25)
            fig.tight_layout()
            fig.savefig("trust_beta_otmf{}.pdf".format("_" + key if key is not None else ""), transparent=True)
            plt.close(fig)

        gd_trust, mal_trust, sel_trust = map(Trust.trust_from_file, [self.good, self.malicious, self.selfish])
        mal_mobile = mal_trust.xs('All Mobile', level='var')
        gd_mobile = gd_trust.xs('All Mobile', level='var')
        sel_mobile = sel_trust.xs('All Mobile', level='var')
        gd_beta_t = beta_trusts(gd_mobile)
        mal_beta_t = beta_trusts(mal_mobile)
        sel_beta_t = beta_trusts(sel_mobile)

        np.random.seed(42)
        mtfm_span = 2
        gd_beta_t['-'] = gd_beta_t['-'].applymap(lambda _: int(2.0 * np.random.random() / 1.5))
        sel_beta_t['-'] = sel_beta_t['-'].applymap(lambda _: int(2.0 * np.random.random() / 1.5))
        mal_beta_t['-'] = mal_beta_t['-'].applymap(lambda _: int(2.0 * np.random.random() / 1.5))

        gd_tp = Trust.generate_node_trust_perspective(gd_mobile)
        mal_tp = Trust.generate_node_trust_perspective(mal_mobile, par=False)
        sel_tp = Trust.generate_node_trust_perspective(sel_mobile)

        gd_mtfm = Trust.generate_mtfm(gd_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)
        mal_mtfm = Trust.generate_mtfm(mal_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)
        sel_mtfm = Trust.generate_mtfm(sel_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)

        gd_mtfm = pd.stats.moments.ewma(gd_mtfm, span=mtfm_span)
        mal_mtfm = pd.stats.moments.ewma(mal_mtfm, span=mtfm_span)
        sel_mtfm = pd.stats.moments.ewma(sel_mtfm, span=mtfm_span)

        plot_beta_mtmf_comparison(gd_beta_t, gd_mtfm, key="fair")
        plot_beta_mtmf_comparison(mal_beta_t, mal_mtfm, key="malicious")
        plot_beta_mtmf_comparison(sel_beta_t, sel_mtfm, key="selfish")

        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)

    def testOutlierGraphs(self):
        columns = {'ADelay': "$Delay$",
                   'ARXP': "$P_{RX}$",
                   'ATXP': "$P_{TX}$",
                   'RXThroughput': "$T^P_{RX}$",
                   'TXThroughput': "$T^P_{TX}$",
                   'PLR': '$PLR$'
                   }
        key_order = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']

        # Plot relative importance of outlier metrics
        required_files = ["MaliciousSelfishMetricFactorsRad.png", "MaliciousSelfishMetricFactors.png"]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        def feature_extractor(df, target):
            data = df.drop(target, axis=1)
            reg = ske.RandomForestRegressor(n_jobs=2, n_estimators=100)
            reg.fit(data, df[target])
            return pd.Series(dict(zip(data.keys(), reg.feature_importances_)))

        def comparer(base, comp, index=0):
            dp_r = (comp / base).reset_index()
            sfet = feature_extractor(dp_r, index)
            return sfet

        # Generate Outlier DataPackage
        with pd.get_store(Tools.in_results('outliers.h5')) as s:
            outliers = s.get('outliers')
            sum_by_weight = outliers.groupby(
                ['bev', u'ADelay', u'ARXP', u'ATXP', u'RXThroughput', u'PLR', u'TXThroughput']).sum().reset_index('bev')
            dp = pd.DataFrame.from_dict({
                                            k: sum_by_weight[sum_by_weight.bev == k]['Delta']
                                            for k in pd.Series(sum_by_weight['bev'].values.ravel()).unique()
                                            })

        # Perform Comparisons
        gm = comparer(dp.good, dp.malicious)[key_order]
        gs = comparer(dp.good, dp.selfish)[key_order]
        ms = comparer(dp.malicious, dp.selfish)[key_order]

        # Bar Chart
        figsize = latexify(columns=1.0, factor=1.2)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        _df = pd.DataFrame.from_dict({"Fair/MPC": gm, "Fair/STS": gs, "MPC/STS": ms}).rename(index=columns)
        ax = _df.plot(kind='bar', position=0.5, ax=ax, legend=False)
        bars = ax.patches
        hatches = ''.join(h * len(_df) for h in 'x/O.')

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticklabels(_df.index, rotation=0)
        ax.set_ylabel("Relative Significance")
        ax.legend(loc='center right', bbox_to_anchor=(1, 1), ncol=4)

        ax = format_axes(ax)
        fig.tight_layout()
        ax.grid(True, color='k', alpha=0.33, ls=':')
        fig.savefig("MaliciousSelfishMetricFactors.png", transparent=True)

        # Radar Base
        with rc_context(rc={'axes.labelsize': 8}):
            figsize = latexify(columns=1, factor=1.2)
            r = radar.radar_factory(len(key_order))
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection='radar')
            ax.plot(r, gm.values, ls='--', marker='x', label="F/M")
            ax.plot(r, gs.values, ls=':', marker='x', label="F/S")
            ax.plot(r, ms.values, ls='-.', marker='x', label="M/S")
            ax.grid(True, color='k', alpha=0.33, ls=':')
            ax.set_varlabels([columns[k] for k in key_order])
            ax.set_ylabel("Relative\nSignificance")
            ax.yaxis.labelpad = -80
            ax.tick_params(direction='out', pad=-100)
            ax.legend(loc=5, mode='expand', bbox_to_anchor=(1.05, 1))
            fig.savefig("MaliciousSelfishMetricFactorsRad.png", transparent=True)

        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)



if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(ThesisLazyDiagrams("testSummaryGraphsForMalGdScenarios"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
