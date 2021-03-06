# coding=utf-8
from __future__ import division

from scripts.publication_scripts import savefig, interpolate_rate_sep, plot_contour_2d, plot_contour_3d, \
    get_mobility_stats, get_emmission_stats, rate_and_ranges, app_rate_from_path, generate_figure_contact_tex, \
    get_separation_stats, saveinput

__author__ = 'bolster'

# coding: utf-8

# # Required Modules and Configurations

# In[1]:

import os
from os.path import expanduser
import itertools
import functools
import warnings
import unittest
import logging
import tempfile
import shutil

logging.basicConfig()

import pandas as pd
import numpy as np
import sklearn.ensemble as ske
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import rc_context
import matplotlib.ticker as plticker
import seaborn as sns

loc_25 = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
fmt_obs_to_mins = plticker.FuncFormatter(lambda x,pos: int(x*10))

import aietes
import aietes.Tools as Tools

import bounos.ChartBuilders as cb
import bounos.Analyses.Trust as Trust
from bounos.ChartBuilders import weight_comparisons, radar, format_axes, _texcol, _texcolhalf, _texcolthird, _texfac

##################
# USING TEMP DIR #
##################
use_temp_dir = False


hatching = False
img_extn = 'pdf'

mobilities = [
    'static',
    'single_mobile',
    'allbut1_mobile',
    'all_mobile'
]

selected_scenarios = map(lambda s: 'bella_'+s, mobilities)


class ThesisLazyDiagrams(unittest.TestCase):
    longMessage = True

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
                self.fail("No file {0}".format(filename))

    def tearDown(self):
        logging.info("Successfully Generated:\n{0}".format(self.generated_files))
        with open("generated_files.txt", 'w') as f:
            f.write('\n'.join(self.generated_files))
        generate_figure_contact_tex(self.generated_files)
        if use_temp_dir:
            shutil.rmtree(self.dirpath, ignore_errors=True)

    def assertFileExists(self, f):
        try:
            self.assertTrue(os.path.isfile(f))
        except AssertionError:
            print f
            raise
        self.generated_files.append(f)

    def test_PhysicalNodeLayout(self):
        # # Graph: Physical Layout of Nodes
        required_files = ["s1_layout." + img_extn]
        #
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        figsize = cb.latexify(columns=_texcol, factor=_texfac)

        base_config = aietes.Simulation.populate_config(
            aietes.Tools.get_config('bella_static.conf'),
            retain_default=True
        )
        texify = lambda t: "${0}_{1}$".format(t[0], t[1])
        node_positions = {texify(k): np.asarray(v['initial_position'], dtype=float) for k, v in
                          base_config['Node']['Nodes'].items() if 'initial_position' in v}
        node_links = {0: [1, 2, 3], 1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 5], 3: [0, 1, 4], 4: [1, 3, 5], 5: [1, 2, 4]}

        fig = cb.plot_nodes(node_positions, figsize=figsize, node_links=node_links, radius=0, scalefree=True,
                            square=True)
        fig.tight_layout(pad=0.3)
        savefig(fig,"s1_layout", transparent=True)
        plt.close(fig)

        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)

    def test_ThroughputLines(self):
        # # Plot Throughput Lines

        required_files = [
            "throughput_sep_lines_static." + img_extn,
            "throughput_sep_lines_all_mobile." + img_extn
        ]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        cb.latexify(columns=_texcol, factor=_texfac)

        for mobility in mobilities:
            df = get_mobility_stats(mobility)
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            for (k, g), ls in zip(df.groupby('separation'), itertools.cycle(["-", "--", "-.", ":"])):
                ax.plot(g.rate, g.throughput, label=k, linestyle=ls)
            ax.legend(loc="upper left")
            ax.set_xlabel("Packet Emission Rate (pps)")
            ax.set_ylabel("Avg. Throughput (bps)")
            fig.tight_layout()
            savefig(fig,"throughput_sep_lines_{0}".format(
                mobility), transparent=True, facecolor='white')
            plt.close(fig)

        for f in required_files:
            self.assertTrue(os.path.isfile(f))
            self.generated_files.append(f)

    def test_MTFMBoxplots(self):
        # # MTFM Boxplots
        required_files = [
            "trust_bella_static_fair." + img_extn,
            "trust_bella_all_mobile_fair." + img_extn,
            "trust_bella_static_malicious." + img_extn,
            "trust_bella_all_mobile_malicious." + img_extn,
            "trust_bella_static_selfish." + img_extn,
            "trust_bella_all_mobile_selfish." + img_extn
        ]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        figsize = cb.latexify(columns=_texcolhalf, factor=_texfac)

        weight_comparisons.plot_mtfm_boxplot(self.good, keyword="fair",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True, prefix="",
                                             extension=img_extn)
        weight_comparisons.plot_mtfm_boxplot(self.malicious, keyword="malicious",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True, prefix="",
                                             extension=img_extn)
        weight_comparisons.plot_mtfm_boxplot(self.selfish, keyword="selfish",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True, prefix="",
                                             extension=img_extn)
        for f in required_files:
            self.assertTrue(os.path.isfile(f), f)
            self.generated_files.append(f)

    def test_PacketStatsGraphs(self):
        # Packet Emissions Graphs for Single Runs
        def write_packet_stats_table(stats, var, prefix, suffix, include_ideal=False):

            stats['error_rate'] = stats.rx_counts / stats.tx_counts
            colmap = {'error_rate': 'Probability of Arrival', 'average_rx_delay': 'Delay(s)'}
            table = stats[['average_rx_delay', 'error_rate']].groupby(level='var').mean().rename(columns=colmap)
            table.index.set_names(var, inplace=True)

            ratio = (stats.rts_counts / stats.rx_counts)
            r_mean = ratio.groupby(level='var').mean()
            table['RTS/Data Ratio'] = r_mean

            table.reset_index(inplace=True)
            if include_ideal:
                table['Ideal Delivery Time(s)'] = table[var] / 1400.0 + 9600.0 / (10000.0)

            tex = table.to_latex(float_format=lambda x: "{0:1.4f}".format(x), index=False, column_format="""
            *{5}{@{\\hspace{1em}}p{0.15\\textwidth} @{\\hspace{1em}}}  """)
            saveinput(tex, "{0}_packet_stats_{1}".format(prefix, suffix))
            print tex

        def plot_packet_stats_for_scenario_containing(scenario_partial):
            statsd = {}
            for store_path in filter(lambda s: scenario_partial in s, rate_and_ranges):
                with pd.get_store(store_path) as s:
                    try:
                        stats = s.get('stats')
                    except KeyError:
                        print store_path
                        print s.keys()
                        raise
                    # Reset Range for packet emission rate
                    stats.index = stats.index.set_levels([
                                                             np.int32((np.asarray(
                                                                 stats.index.levels[0].astype(np.float64)) * 100)),
                                                             # Var
                                                             stats.index.levels[1].astype(np.int32)  # Run
                                                         ] + (stats.index.levels[2:])
                                                         )
                    statsd[app_rate_from_path(store_path)] = stats.copy()

            df = pd.concat(statsd.values(), keys=statsd.keys(), names=['rate', 'separation'] + stats.index.names[1:])
            base_df = df.reset_index().sort_values(by=['rate', 'separation']).set_index(
                ['rate', 'separation', 'run', 'node'])

            rename_labels = {"rx_counts": "Throughput",
                             "tx_counts": "Offered Load",
                             "enqueued": "Enqueued Packets",
                             "collisions": "Collisions"}

            figsize = cb.latexify(columns=_texcolhalf, factor=_texfac)

            # Emission Stats

            stats = get_emmission_stats(base_df, separation=100)
            var = "Packet Emission Rate (pps)"

            write_packet_stats_table(stats, var, 'emission', scenario_partial)

            fig = cb.performance_summary_for_var(stats,
                                                 var=var, title=False,
                                                 rename_labels=rename_labels,
                                                 hide_annotations=True, figsize=figsize)
            savefig(fig, "emission_throughput_performance_" + scenario_partial, img_extn)

            fig = cb.probability_of_timely_arrival(stats, var=var, title=False, figsize=figsize)
            savefig(fig, "emission_prod_breakdown_" + scenario_partial, img_extn)

            fig = cb.average_delays_across_variation(stats, var=var, title=False, figsize=figsize)
            savefig(fig, "emission_delay_variation_" + scenario_partial, img_extn)

            fig = cb.rts_ratio_across_variation(stats, var=var, title=False, figsize=figsize)
            savefig(fig, "emission_rts_ratio_" + scenario_partial, img_extn)

            # Separation Stats

            stats = get_separation_stats(base_df, emission=0.02)
            var = "Initial Node Separation (m)"

            write_packet_stats_table(stats, var, 'separation', scenario_partial, include_ideal=True)

            fig = cb.performance_summary_for_var(stats,
                                                 var=var, title=False,
                                                 rename_labels=rename_labels,
                                                 hide_annotations=True, figsize=figsize)
            savefig(fig, "separation_throughput_performance_" + scenario_partial, img_extn)

            fig = cb.probability_of_timely_arrival(stats, var=var, title=False, figsize=figsize)
            savefig(fig, "separation_prod_breakdown_" + scenario_partial, img_extn)

            fig = cb.average_delays_across_variation(stats, var=var, title=False, figsize=figsize)
            savefig(fig, "separation_delay_variation_" + scenario_partial, img_extn)

            fig = cb.rts_ratio_across_variation(stats, var=var, title=False, figsize=figsize)
            savefig(fig, "separation_rts_ratio_" + scenario_partial, img_extn)

            plt.close('all')

        required_file_prefixes = [
            "throughput_performance",
            "prod_breakdown",
            "delay_variation",
            "rts_ratio"
        ]
        supra_prefixes = ['emission', 'separation']
        for scenario in selected_scenarios:
            for supra in supra_prefixes:
                for prefix in required_file_prefixes:
                    Tools.remove("{0}_{1}_{2}.{3}".format(
                        supra,
                        prefix,
                        scenario,
                        img_extn
                    ))
            plot_packet_stats_for_scenario_containing(scenario)
            for supra in supra_prefixes:
                for prefix in required_file_prefixes:
                    f = "{0}_{1}_{2}.{3}".format(
                        supra,
                        prefix,
                        scenario,
                        img_extn
                    )
                    self.assertTrue(os.path.isfile(f))
                    self.generated_files.append(f)

    def test_RateRangePlots(self):
        # Plot the Fancy 2d/3d graphs of rate/range/performance

        def rate_range_plots_per_scenario(scenario_partial):
            statsd = {}
            for store_path in filter(lambda s: scenario_partial in s, rate_and_ranges):
                with pd.get_store(store_path) as s:
                    try:
                        stats = s.get('stats')
                    except KeyError:
                        print store_path
                        print s.keys()
                        raise
                    # Reset Range for packet emission rate
                    stats.index = stats.index.set_levels([
                                                             np.int32((np.asarray(
                                                                 stats.index.levels[0].astype(np.float64)) * 100)),
                                                             # Var
                                                             stats.index.levels[1].astype(np.int32)  # Run
                                                         ] + (stats.index.levels[2:])
                                                         )
                    statsd[app_rate_from_path(store_path)] = stats.copy()

            df = pd.concat(statsd.values(), keys=statsd.keys(), names=['rate', 'separation'] + stats.index.names[1:])
            base_df = df.reset_index().sort(['rate', 'separation']).set_index(['rate', 'separation', 'run', 'node'])

            df = base_df.groupby(level=['rate', 'separation']).mean().reset_index()

            norm = lambda df: (df - np.nanmin(df)) / (np.nanmax(df) - np.nanmin(df))

            df['average_rx_delay_norm'] = 1 - norm(df.average_rx_delay)
            df['throughput_norm'] = norm(df.throughput)
            df['co_norm'] = df.average_rx_delay_norm * df.throughput_norm
            df = df.set_index(['rate', 'separation']).dropna()
            df['tdivdel'] = (df.throughput / df.average_rx_delay)
            df.reset_index(inplace=True)

            figsize = cb.latexify(columns=_texcol * 2, factor=_texfac)

            xt, yt, zt, Xt, Yt = interpolate_rate_sep(df.dropna(), "throughput")
            fig = plot_contour_2d(xt, yt, zt, Xt, Yt, "Throughput (bps)", figsize=figsize)
            savefig(fig, "throughput_2d_" + scenario_partial, img_extn)

            xd, yd, zd, Xd, Yd = interpolate_rate_sep(df.dropna(), "average_rx_delay")
            fig = plot_contour_2d(xd, yd, zd, Xd, Yd, "Average Delay (s)", figsize=figsize)
            savefig(fig, "delay_2d_" + scenario_partial, img_extn)

            xd, yd, zd, Xd, Yd = interpolate_rate_sep(df, "tdivdel")
            fig = plot_contour_2d(xd, yd, zd, Xd, Yd, "Throughput Delay Ratio", figsize=figsize)
            savefig(fig, "2d_ratio_" + scenario_partial, img_extn)

            fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
            savefig(fig, "3d_ratio_" + scenario_partial, img_extn, transparent=True, facecolor='white')

            xd, yd, zd, Xd, Yd = interpolate_rate_sep(df, "co_norm")
            fig = plot_contour_2d(xd, yd, zd, Xd, Yd, "Normalised Throughput Delay Product", norm=True, figsize=figsize)
            savefig(fig, "2d_normed_product_" + scenario_partial, img_extn)

            fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
            savefig(fig, "3d_normed_product_" + scenario_partial, img_extn, transparent=True, facecolor='white')

        required_file_prefixes = [
            "throughput_2d_",
            "delay_2d_",
            "2d_ratio_",
            "3d_ratio_",
            "2d_normed_product_",
            "3d_normed_product_"
        ]
        for scenario in selected_scenarios:
            for prefix in required_file_prefixes:
                Tools.remove("{0}{1}.{2}".format(
                    prefix,
                    scenario,
                    img_extn
                ))
            rate_range_plots_per_scenario(scenario)
            for prefix in required_file_prefixes:
                f = "{0}{1}.{2}".format(
                    prefix,
                    scenario,
                    img_extn
                )
                self.assertFileExists(f)

    def test_WeightComparisons(self):
        # # Weight Comparisons

        required_files = [
            "trust_bella_static_emph_ADelay_BadMouthingPowerControl." + img_extn,
            "trust_bella_static_emph_ATXP_BadMouthingPowerControl." + img_extn,
            "trust_bella_static_emph_RXThroughput_BadMouthingPowerControl." + img_extn,
            "trust_bella_static_emph_TXThroughput_BadMouthingPowerControl." + img_extn,
            "trust_bella_all_mobile_emph_ADelay_BadMouthingPowerControl." + img_extn,
            "trust_bella_all_mobile_emph_ARXP_BadMouthingPowerControl." + img_extn,
            "trust_bella_all_mobile_emph_ATXP_BadMouthingPowerControl." + img_extn,
            "trust_bella_all_mobile_emph_RXThroughput_BadMouthingPowerControl." + img_extn,
            "trust_bella_all_mobile_emph_TXThroughput_BadMouthingPowerControl." + img_extn,
            "trust_bella_static_emph_ADelay_SelfishTargetSelection." + img_extn,
            "trust_bella_static_emph_ATXP_SelfishTargetSelection." + img_extn,
            "trust_bella_static_emph_RXThroughput_SelfishTargetSelection." + img_extn,
            "trust_bella_static_emph_TXThroughput_SelfishTargetSelection." + img_extn,
            "trust_bella_all_mobile_emph_ADelay_SelfishTargetSelection." + img_extn,
            "trust_bella_all_mobile_emph_ARXP_SelfishTargetSelection." + img_extn,
            "trust_bella_all_mobile_emph_ATXP_SelfishTargetSelection." + img_extn,
            "trust_bella_all_mobile_emph_RXThroughput_SelfishTargetSelection." + img_extn,
            "trust_bella_all_mobile_emph_TXThroughput_SelfishTargetSelection." + img_extn
        ]
        for f in required_files:
            Tools.remove(f)

        figsize = cb.latexify(columns=_texcolhalf, factor=_texfac)

        for s in selected_scenarios:
            try:
                weight_comparisons.plot_weight_comparisons(self.good, self.malicious,
                                                           malicious_behaviour="BadMouthingPowerControl",
                                                           s=s, figsize=figsize, show_title=False,
                                                           labels=["Fair", "Malicious"],
                                                           prefix="", extension=img_extn
                                                           )
                weight_comparisons.plot_weight_comparisons(self.good, self.selfish,
                                                           malicious_behaviour="SelfishTargetSelection",
                                                           s=s, figsize=figsize, show_title=False,
                                                           labels=["Fair", "Selfish"],
                                                           prefix="", extension=img_extn
                                                           )
                weight_comparisons.plot_weight_comparisons(self.good, self.selfish,
                                                           malicious_behaviour="SelfishTargetSelection",
                                                           s=s, figsize=figsize, show_title=False,
                                                           labels=["Fair", "Selfish"],
                                                           prefix="", extension=img_extn
                                                           )
            except KeyError:
                warnings.warn("Scenario {0} not in trust run, skipping".format(s))
        for f in required_files:
            self.assertFileExists(f)

    def test_SummaryGraphsForMalGdScenarios(self):
        # # Summary Graphs with malicious, selfish and fair scenarios
        required_files = ["trust_beta_otmf_fair." + img_extn,
                          "trust_beta_otmf_malicious." + img_extn,
                          "trust_beta_otmf_selfish." + img_extn]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        cb.latexify(columns=_texcol, factor=_texfac)

        def beta_trusts(trust, length=4096):
            # TODO This should be optimised to not use the same dataframe
            trust['+'] = (trust.TXThroughput / length) * (1 - trust.PLR)
            trust['-'] = (trust.TXThroughput / length) * trust.PLR
            beta_trust = trust[['+', '-']].unstack(level='target')
            trust.drop(['+', '-'], axis=1, inplace=True)
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
            figsize = cb.latexify(columns=_texcolhalf, factor=_texfac)
            fig, ax = plt.subplots(1, 1, sharey=True, figsize=figsize)
            ax.plot(beta.xs('n0', level='observer')['n1'].values, label="Hermes")
            ax.plot(otmf.xs('n0', level='observer')['n1'].values, label="OTMF")
            ax.plot(mtfm.values, label="MTFM")
            ax.set_ylim((0, 1))
            ax.set_ylabel("Trust Value".format(key))
            ax.set_xlabel("Mission Time (mins)")
            ax = format_axes(ax)
            ax.legend(loc='lower center', ncol=3)
            ax.yaxis.set_major_locator(loc_25)
            ax.xaxis.set_major_formatter(fmt_obs_to_mins)
            fig.tight_layout()
            savefig(fig,"trust_beta_otmf{0}".format(
                "_" + key if key is not None else "",
                ), transparent=True)
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
            self.assertFileExists(f)

    def test_InvertedSummaryGraphsForMalGdScenarios(self):
        # # Summary Graphs with malicious, selfish and fair scenarios
        required_files = ["trust_beta_otmf_mtfm_boxes."+img_extn,]
        for f in required_files:
            try:
                os.remove(f)
            except:
                pass

        figsize = cb.latexify(columns=_texcol, factor=_texfac)

        def beta_trusts(trust, length=4096):
            # TODO This should be optimised to not use the same dataframe
            trust['+'] = (trust.TXThroughput / length) * (1 - trust.PLR)
            trust['-'] = (trust.TXThroughput / length) * trust.PLR
            beta_trust = trust[['+', '-']].unstack(level='target')
            trust.drop(['+', '-'], axis=1, inplace=True)
            return beta_trust

        def beta_calcs(beta_trust, observer='n0', target='n1'):
            beta_trust = pd.stats.moments.ewma(beta_trust, span=2)
            beta_t_confidence = lambda s, f: 1 - np.sqrt((12 * s * f) / ((s + f + 1) * (s + f) ** 2))
            beta_t = lambda s, f: s / (s + f)
            otmf_t = lambda s, f: 1 - np.sqrt(
                ((((beta_t(s, f) - 1) ** 2) / 2) + (((beta_t_confidence(s, f) - 1) ** 2) / 9))) / np.sqrt(
                (1 / 2) + (1 / 9))
            beta_vals = beta_trust.apply(lambda r: beta_t(r['+'], r['-']), axis=1)
            otmf_vals = beta_trust.apply(lambda r: otmf_t(r['+'], r['-']), axis=1)
            if observer:
                beta_vals = beta_vals.xs(observer, level='observer')
                otmf_vals = otmf_vals.xs(observer, level='observer')
            if target:
                beta_vals = beta_vals[target]
                otmf_vals = otmf_vals[target]
            return beta_vals, otmf_vals

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
        sel_beta_t['-'] = sel_beta_t['-'].applymap(lambda _: int(2.0 * np.random.random() / 1.8))
        mal_beta_t['-'] = mal_beta_t['-'].applymap(lambda _: int(2.0 * np.random.random() / 1.8))

        gd_tp = Trust.generate_node_trust_perspective(gd_mobile)
        mal_tp = Trust.generate_node_trust_perspective(mal_mobile, par=False)
        sel_tp = Trust.generate_node_trust_perspective(sel_mobile)

        gd_mtfm = Trust.generate_mtfm(gd_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)
        mal_mtfm = Trust.generate_mtfm(mal_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)
        sel_mtfm = Trust.generate_mtfm(sel_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)

        gd_mtfm = pd.stats.moments.ewma(gd_mtfm, span=mtfm_span)
        mal_mtfm = pd.stats.moments.ewma(mal_mtfm, span=mtfm_span)
        sel_mtfm = pd.stats.moments.ewma(sel_mtfm, span=mtfm_span)

        # Fix MTFM inclusion of Var multiindex level
        gd_mtfm.index = gd_mtfm.index.droplevel('var')
        mal_mtfm.index = mal_mtfm.index.droplevel('var')
        sel_mtfm.index = sel_mtfm.index.droplevel('var')

        gd_beta,  gd_otmf  = beta_calcs(gd_beta_t)
        mal_beta, mal_otmf = beta_calcs(mal_beta_t)
        sel_beta, sel_otmf = beta_calcs(sel_beta_t)

        intermediate = {
            'Hermes':{
                'Fair':gd_beta,
                'MPC':mal_beta,
                'STS':sel_beta
            },
            'OTMF':{
                'Fair':gd_otmf,
                'MPC':mal_otmf,
                'STS':sel_otmf
            },
            'MTFM':{
                'Fair':gd_mtfm,
                'MPC':mal_mtfm,
                'STS':sel_mtfm
            }
        }

        df = pd.DataFrame.from_dict(Tools.tuplify_multi_dict(intermediate))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        df.index = df.index.droplevel('run')

        df = df.reindex_axis(['Hermes', 'OTMF', 'MTFM'], axis=1, level=0)
        mdf = pd.melt(df)
        mdf.columns = ['Trust Framework', 'Scenario', 'Trust Assessment']
        with sns.axes_style(cb._latexify_rcparams):
            sns.boxplot(data=mdf, x='Trust Framework', y='Trust Assessment', hue='Scenario', ax=ax,
                        showfliers=False, whis=1)
            ax.set_ylim(0, 1)
            ax = format_axes(ax)
            savefig(fig, "trust_beta_otmf_mtfm_boxes")
        for f in required_files:
            self.assertFileExists(f)

    def test_CommsOutlierGraphs(self):
        # Plot relative importance of outlier metrics
        required_files = ["MaliciousSelfishMetricFactorsRad."+img_extn,
                          "MaliciousSelfishMetricFactors."+img_extn]
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

        key_order = Tools.key_order[0:-3]

        # Generate Outlier DataPackage
        with pd.get_store(Tools.in_results('outliers.h5')) as s:
            outliers = s.get('outliers')
            sum_by_weight = outliers.groupby(
                ['bev', u'ADelay', u'ARXP', u'ATXP', u'RXThroughput', u'PLR', u'TXThroughput']).sum().reset_index(
                'bev')
            dp = pd.DataFrame.from_dict({
                                            k: sum_by_weight[sum_by_weight.bev == k]['Delta']
                                            for k in pd.Series(sum_by_weight['bev'].values.ravel()).unique()
                                            })

        # Perform Comparisons
        gm = comparer(dp.good, dp.malicious)[key_order]
        gs = comparer(dp.good, dp.selfish)[key_order]
        ms = comparer(dp.malicious, dp.selfish)[key_order]

        # Bar Chart
        figsize = cb.latexify(columns=_texcol, factor=_texfac)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        _df = pd.DataFrame.from_dict({"Fair/MPC": gm, "Fair/STS": gs, "MPC/STS": ms}).rename(
            index=Tools.metric_rename_dict)
        ax = _df.plot(kind='bar', position=0.5, ax=ax, legend=False)
        bars = ax.patches

        if hatching:
            hatches = ''.join(h * len(_df) for h in 'x/O.')
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
        ax.set_xticklabels(_df.index, rotation=0)
        ax.set_ylabel("Relative Significance")
        ax.legend(loc='center right', bbox_to_anchor=(1, 1), ncol=4)

        ax = format_axes(ax)
        fig.tight_layout()
        ax.grid(True, color='k', alpha=0.33, ls=':')
        savefig(fig, "MaliciousSelfishMetricFactors", transparent=True)

        # Radar Base
        with rc_context(rc={'axes.labelsize': 8}):
            figsize = cb.latexify(columns=_texcol, factor=_texfac)
            r = radar.radar_factory(len(key_order))
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection='radar')
            ax.plot(r, gm.values, ls='--', marker='x', label="F/M")
            ax.plot(r, gs.values, ls=':', marker='x', label="F/S")
            ax.plot(r, ms.values, ls='-.', marker='x', label="M/S")
            ax.grid(True, color='k', alpha=0.33, ls=':')
            ax.set_varlabels([Tools.metric_rename_dict[k] for k in key_order])
            ax.set_ylabel("Relative\nSignificance")
            ax.yaxis.labelpad = -80
            ax.tick_params(direction='out', pad=-100)
            ax.legend(loc=5, mode='expand', bbox_to_anchor=(1.05, 1))
            savefig(fig, "MaliciousSelfishMetricFactorsRad", transparent=True)

        for f in required_files:
            self.assertFileExists(f)

