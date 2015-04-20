from __future__ import division

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
logging.basicConfig()

import pandas as pd
import numpy as np
from collections import OrderedDict

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.ticker as plticker
loc_25 = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals

import aietes
import aietes.Tools as Tools
import bounos.ChartBuilders as cb
import bounos.Analyses.Trust as Trust
from bounos.ChartBuilders import weight_comparisons

import scipy.interpolate as interpolate
# pylab.rcParams['figure.figsize'] = 16, 12

in_results = functools.partial(os.path.join, Tools._results_dir)
app_rate_from_path = lambda s: float(".".join(s.split('-')[2].split('.')[0:-1]))
scenario_map = dict(zip(
    [u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static'],
    ['All Mobile', '$n_1$ Static', '$n_1$ Mobile', 'All Static']
))
scenario_order = list(reversed([u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static']))

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6
cb.latexify(columns=2, factor=0.55)

# # Data File Acquistion

# In[4]:

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing
result_h5s_by_latest = sorted(filter(lambda p: os.path.basename(p).endswith("h5"),
                                     map(lambda p: os.path.abspath(os.path.join(Tools._results_dir, p)),
                                         os.listdir(Tools._results_dir))), key=lambda f: os.path.getmtime(f))
rate_and_ranges = filter(lambda p: os.path.basename(p).startswith("CommsRateAndRangeTest"),
                         result_h5s_by_latest)
app_rates = map(app_rate_from_path, rate_and_ranges)



def interpolate_rate_sep(df, key):
    X, Y, Z = df.rate, df.separation, df[key]

    xi = np.linspace(X.min(), X.max(), 16)
    yi = np.linspace(Y.min(), Y.max(), 16)
    # VERY IMPORTANT, to tell matplotlib how is your data organized
    zi = interpolate.griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='linear')
    return xi, yi, zi, X, Y


def plot_contour_pair(xi, yi, zi):
    fig = plt.figure(figsize=(2 * w, golden_mean * w * 2))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, color='k')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    xig, yig = np.meshgrid(xi, yi)

    surf = ax.plot_surface(xig, yig, zi, linewidth=0)
    return fig


def plot_contour_3d(xi, yi, zi, rot=120, labels=None):
    fig = plt.figure(figsize=(w, golden_mean * w))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # Normalise Z
    zi_norm = zi / np.nanmax(zi)
    xig, yig = np.meshgrid(xi, yi)

    if rot < 90:
        zoffset = np.nanmin(zi)
        xoffset = np.nanmin(xig)
        yoffset = np.nanmin(yig)
    elif rot < 180:
        zoffset = np.nanmin(zi)
        xoffset = np.nanmax(xig)
        yoffset = np.nanmin(yig)
    else:
        zoffset = np.nanmin(zi)
        xoffset = np.nanmax(xig)
        yoffset = np.nanmax(yig)

    ax.plot_surface(xig, yig, zi, rstride=1, cstride=1, alpha=0.45, facecolors=cm.coolwarm(zi_norm), linewidth=1,
                    antialiased=True)
    cset = ax.contour(xig, yig, zi, zdir='z', offset=zoffset, linestyles='dashed', cmap=cm.coolwarm)
    cset = ax.contour(xig, yig, zi, zdir='x', offset=xoffset, cmap=cm.coolwarm)
    cset = ax.contour(xig, yig, zi, zdir='y', offset=yoffset, cmap=cm.coolwarm)
    ax.view_init(30, rot)

    if labels is not None:
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'])
        ax.set_zlabel(labels['z'])

    fig.tight_layout()
    return fig


def plot_lines_of_throughput(df):


    return fig


def plot_contour_2d(xi, yi, zi, X=[], Y=[], var=None, norm=False):
    fig = plt.figure(figsize=(w, golden_mean * w), facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    xig, yig = np.meshgrid(xi, yi)
    x_min, y_min = map(np.nanmin, (xi, yi))
    x_max, y_max = map(np.nanmax, (xi, yi))
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([x_min, x_max])
    ax.set_xlabel("Packet Emission Rate (pps)")
    ax.set_ylabel("Average Node Separation (m)")

    if norm:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(abs(zi))
        vmax = np.nanmax(abs(zi))

    cset = ax.contourf(zi, alpha=0.75,  # hatches=['+','x','-', '/', '\\', '//'],
                       cmap=plt.get_cmap('hot_r'),
                       vmin=vmin, vmax=vmax,
                       extent=[x_min, x_max, y_min, y_max]
                       )
    cbar = fig.colorbar(cset)
    if var is not None:
        # ax.set_title("{} with varying Packet Emission and Node Separations".format(var))
        cbar.set_label(var)

    if len(X) and len(Y):
        ax.scatter(X, Y, color='k', marker='.')

    # ax.clabel(cset,inline=1)
    return fig


def plot_all_funky_stuff(sdf, mobility):
    xt, yt, zt, Xt, Yt = interpolate_rate_sep(sdf.dropna(), "throughput")
    fig = plot_contour_2d(xt, yt, zt, Xt, Yt, "Per Node Avg. Throughput (bps)")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/throughput_2d_{}.pdf".format(mobility))

    xd, yd, zd, Xd, Yd = interpolate_rate_sep(sdf.dropna(), "average_rx_delay")
    fig = plot_contour_2d(xd, yd, zd, Xd, Yd, "Average Delay (s)")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/delay_2d_{}.pdf".format(mobility))

    xd, yd, zd, Xd, Yd = interpolate_rate_sep(sdf, "tdivdel")
    fig = plot_contour_2d(xd, yd, zd, Xd, Yd, "Throughput Delay Ratio")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/2d_ratio_{}.pdf".format(mobility))

    fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
    fig.tight_layout(pad=0.1)
    fig.savefig("img/3d_ratio_{}.pdf".format(mobility), transparent=True, facecolor='white')

    xd, yd, zd, Xd, Yd = interpolate_rate_sep(sdf, "co_norm")
    fig = plot_contour_2d(xd, yd, zd, Xd, Yd, "Normalised Throughput Delay Product", norm=True)
    fig.tight_layout(pad=0.1)
    fig.savefig("img/2d_normed_product_{}.pdf".format(mobility))

    fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
    fig.tight_layout(pad=0.1)
    fig.savefig("img/3d_normed_product_{}.pdf".format(mobility), transparent=True, facecolor='white')

    fig = plot_lines_of_throughput(sdf)
    fig.tight_layout(pad=0.1)
    fig.savefig("img/throughput_sep_lines_{}.pdf".format(mobility), transparent=True, facecolor='white')


def get_mobility_stats(mobility):
    statsd = OrderedDict()
    trustd = OrderedDict()
    norm = lambda df: (df - np.nanmin(df)) / (np.nanmax(df) - np.nanmin(df))
    rate_and_ranges = filter(
        lambda p: os.path.basename(p).startswith("CommsRateAndRangeTest-bella_{}".format(mobility)),
        result_h5s_by_latest)
    if not rate_and_ranges:
        raise ValueError("No Entries with mobility {}".format(mobility))
    for store_path in sorted(rate_and_ranges):
        with pd.get_store(store_path) as s:
            stats = s.get('stats')
            trust = s.get('trust')
            # Reset Range for packet emission rate
            stats.index = stats.index.set_levels([
                                                     np.int32(
                                                         (np.asarray(stats.index.levels[0].astype(np.float64)) * 100)),
                                                     # Var
                                                     stats.index.levels[1].astype(np.int32)  # Run
                                                 ] + (stats.index.levels[2:])
                                                 )

            statsd[app_rate_from_path(store_path)] = stats.copy()
            trustd[app_rate_from_path(store_path)] = trust.copy()

    def df_for_rates_and_sep(d, last_names):
        df = pd.concat(d.values(), keys=d.keys(), names=['rate', 'separation'] + last_names[1:])

        return df

    sdf = df_for_rates_and_sep(statsd, stats.index.names)
    tdf = df_for_rates_and_sep(trustd, trust.index.names)
    sdf['throughput'] = sdf.throughput / 3600.0
    sdf = sdf.groupby(level=['rate', 'separation']).mean().reset_index()

    sdf['average_rx_delay_norm'] = 1 - norm(sdf.average_rx_delay)
    sdf['throughput_norm'] = norm(sdf.throughput)
    sdf['co_norm'] = sdf.average_rx_delay_norm * sdf.throughput_norm
    sdf = sdf.set_index(['rate', 'separation'])
    sdf['tdivdel'] = sdf.throughput / (sdf.average_rx_delay)
    sdf.reset_index(inplace=True)
    return sdf


class TrustCom(unittest.TestCase):
    def setUp(self):

        os.chdir(expanduser("~/src/thesis/papers/active/15_TrustCom/"))

        if not os.path.exists("img"):
            os.makedirs("img")
        if not os.path.exists("input"):
            os.makedirs("input")


        # # Plot Style Config

        _boxplot_kwargs = {
            'showmeans': True,
            'showbox': False,
            'widths': 0.2,
            'linewidth': 2
        }

        self.malicious = "MaliciousBadMouthingPowerControlTrustMedianTests-0.025-3-2015-02-19-23-27-01.h5"
        self.good = "TrustMedianTests-0.025-3-2015-02-19-23-29-39.h5"
        self.selfish = "MaliciousSelfishTargetSelectionTrustMedianTests-0.025-3-2015-03-29-19-32-36.h5"
        self.generated_files = []

        for file in [self.good, self.malicious, self.selfish]:
            if not os.path.isfile(Tools.in_results(file)):
                self.fail("No file {}".format(file))

    def tearDown(self):
        logging.info("Successfully Generated:\n{}".format(self.generated_files))

    def testPhysicalNodeLayout(self):
        # # Graph: Physical Layout of Nodes
        required_files = ["s1_layout.pdf"]
        #
        for f in required_files:
            try:
                os.remove(os.path.join("img", f))
            except:
                pass

        cb.latexify(columns=0.5, factor=0.5)

        base_config = aietes.Simulation.populate_config(
            aietes.Tools.get_config('bella_static.conf'),
            retain_default=True
        )
        texify = lambda t: "${}_{}$".format(t[0], t[1])
        node_positions = {texify(k): np.asarray(v['initial_position'], dtype=float) for k, v in
                          base_config['Node']['Nodes'].items() if v.has_key('initial_position')}
        node_links = {0: [1, 2, 3], 1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 5], 3: [0, 1, 4], 4: [1, 3, 5], 5: [1, 2, 4]}
        reload(cb)
        # _=cb.plot_positions(node_positions, bounds=base_config.Environment.shape)
        fig = cb.plot_nodes(node_positions, figsize=(4,1.6), node_links=node_links, radius=3, scalefree=True, square=False)
        fig.tight_layout(pad=0.3)
        fig.savefig("img/s1_layout.pdf", transparent=True)
        plt.close(fig)

        for f in required_files:
            self.assertTrue(os.path.isfile(os.path.join("img",f)))
            self.generated_files.append(f)

    def testThroughputLines(self):
        # # Plot Throughput Lines
        required_files = [
            "throughput_sep_lines_static.pdf",
            "throughput_sep_lines_all_mobile.pdf"
        ]
        for f in required_files:
            try:
                os.remove(os.path.join("img", f))
            except:
                pass

        cb.latexify(columns=0.5, factor=0.5)

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
            fig.savefig("img/throughput_sep_lines_{}.pdf".format(mobility), transparent=True, facecolor='white')
            plt.close(fig)

        for f in required_files:
            self.assertTrue(os.path.isfile(os.path.join("img",f)))
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
                os.remove(os.path.join("img", f))
            except:
                pass

        figsize = cb.latexify(columns=0.5, factor=0.5)

        selected_scenarios = ['bella_all_mobile', 'bella_static']
        weight_comparisons.plot_mtfm_boxplot(self.good, keyword="fair",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True)
        weight_comparisons.plot_mtfm_boxplot(self.malicious, keyword="malicious",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True)
        weight_comparisons.plot_mtfm_boxplot(self.selfish, keyword="selfish",
                                             s=selected_scenarios, figsize=figsize,
                                             xlabel=False, dropnet=True)
        for f in required_files:
            self.assertTrue(os.path.isfile(os.path.join("img",f)), f)
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
                os.remove(os.path.join("img", f))
            except:
                pass

        figsize = cb.latexify(columns=0.5, factor=0.5)

        selected_scenarios = ['bella_all_mobile', 'bella_static']
        for s in selected_scenarios:
            weight_comparisons.plot_weight_comparisons(self.good, self.malicious,
                                                       malicious_behaviour="BadMouthingPowerControl",
                                                       s=s, figsize=figsize, show_title=False,
                                                       labels=["Fair","Malicious"]
                                                       )
            weight_comparisons.plot_weight_comparisons(self.good, self.selfish,
                                                       malicious_behaviour="SelfishTargetSelection",
                                                       s=s, figsize=figsize, show_title=False,
                                                       labels=["Fair","Selfish"]
                                                       )
        for f in required_files:
            self.assertTrue(os.path.isfile(os.path.join("img",f)))
            self.generated_files.append(f)

    def testSummaryGraphsForMalGdScenarios(self):
        # # Summary Graphs with malicious, selfish and fair scenarios
        required_files = ["trust_beta_otmf_fair.pdf", "trust_beta_otmf_malicious.pdf", "trust_beta_otmf_selfish.pdf"]
        for f in required_files:
            try:
                os.remove(os.path.join("img", f))
            except:
                pass

        cb.latexify(columns=0.5, factor=0.5)

        def beta_trusts(trust, length=4096):
            trust['+'] = (trust.TXThroughput/length)*(1-trust.PLR)
            trust['-'] = (trust.TXThroughput/length)*(trust.PLR)
            beta_trust = trust[['+','-']].unstack(level='target')
            return beta_trust

        def beta_calcs(beta_trust):
            beta_trust=pd.stats.moments.ewma(beta_trust, span=2)
            beta_t_confidence = lambda s,f: 1-np.sqrt((12*s*f)/((s+f+1)*(s+f)**2))
            beta_t = lambda s,f: s/(s+f)
            otmf_T = lambda s,f: 1-np.sqrt(((((beta_t(s,f)-1)**2)/2)+(((beta_t_confidence(s,f) - 1)**2)/9)))/np.sqrt((1/2)+(1/9))
            beta_vals = beta_trust.apply(lambda r: beta_t(r['+'],r['-']), axis=1)
            otmf_vals = beta_trust.apply(lambda r: otmf_T(r['+'],r['-']), axis=1)
            return beta_vals, otmf_vals

        def plot_beta_mtmf_comparison(beta_trust, mtfm, key):

            beta, otmf = beta_calcs(beta_trust)
            fig, ax = plt.subplots(1, 1, sharey=True)
            ax.plot(beta.xs('n0',level='observer')['n1'], label="Hermes", linestyle='--')
            ax.plot(otmf.xs('n0',level='observer')['n1'], label="OTMF", linestyle=':')
            ax.plot(mtfm, label="MTFM")
            ax.set_ylim((0,1))
            ax.set_ylabel("Trust Value".format(key))
            ax.set_xlabel("Observation")
            ax.legend(loc='lower center', ncol=3)
            ax.axhline(mtfm.mean(), color="r", linestyle='-.')
            ax.yaxis.set_major_locator(loc_25)
            fig.tight_layout()
            fig.savefig("img/trust_beta_otmf{}.pdf".format("_"+key if key is not None else ""),transparent=True)
            plt.close(fig)


        gd_trust, mal_trust, sel_trust = map(Trust.trust_from_file,[self.good,self.malicious, self.selfish])
        mal_mobile = mal_trust.xs('All Mobile',level='var')
        gd_mobile = gd_trust.xs('All Mobile',level='var')
        sel_mobile = sel_trust.xs('All Mobile',level='var')
        gd_beta_t = beta_trusts(gd_mobile)
        mal_beta_t = beta_trusts(mal_mobile)
        sel_beta_t = beta_trusts(sel_mobile)

        np.random.seed(42)
        mtfm_span = 2
        gd_beta_t['-'] = gd_beta_t['-'].applymap(lambda _: int(2.0*np.random.random()/1.5))
        sel_beta_t['-'] = sel_beta_t['-'].applymap(lambda _: int(2.0*np.random.random()/1.5))
        mal_beta_t['-'] = mal_beta_t['-'].applymap(lambda _: int(2.0*np.random.random()/1.5))

        gd_tp = Trust.generate_node_trust_perspective(gd_mobile)
        mal_tp = Trust.generate_node_trust_perspective(mal_mobile)
        sel_tp = Trust.generate_node_trust_perspective(sel_mobile)

        gd_mtfm = Trust.generate_mtfm(gd_tp, 'n0', 'n1',['n2','n3'],['n4','n5']).sum(axis=1)
        mal_mtfm = Trust.generate_mtfm(mal_tp, 'n0', 'n1',['n2','n3'],['n4','n5']).sum(axis=1)
        sel_mtfm = Trust.generate_mtfm(sel_tp, 'n0', 'n1',['n2','n3'],['n4','n5']).sum(axis=1)

        gd_mtfm =pd.stats.moments.ewma(gd_mtfm, span=mtfm_span)
        mal_mtfm =pd.stats.moments.ewma(mal_mtfm, span=mtfm_span)
        sel_mtfm =pd.stats.moments.ewma(sel_mtfm, span=mtfm_span)
        plot_beta_mtmf_comparison(gd_beta_t, gd_mtfm, key="fair")
        plot_beta_mtmf_comparison(mal_beta_t, mal_mtfm, key="malicious")
        plot_beta_mtmf_comparison(sel_beta_t, sel_mtfm, key="selfish")

        for f in required_files:
            self.assertTrue(os.path.isfile(os.path.join("img",f)))
            self.generated_files.append(f)
