# coding=utf-8

from __future__ import division, print_function

import tempfile
import unittest
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import palettable.colorbrewer.qualitative as cmap_qual
import palettable.colorbrewer.sequential as cmap_seq
import sklearn.ensemble as ske
from scipy.stats.stats import pearsonr
import cartopy.crs as ccrs
import matplotlib.ticker as mtick

import cartopy.feature as cfeature


import matplotlib.pyplot as plt
from oceans.colormaps import cm
import iris.plot as iplt

from oceans.ff_tools import wrap_lon180

from oceans.datasets import woa_subset, woa_profile
import os
from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
                                   LATITUDE_FORMATTER)

LAND = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face',
                                    facecolor=cfeature.COLORS['land'])

from aietes.Tools import *
from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.Analyses.Weight import target_weight_feature_extractor, \
    generate_weighted_trust_perspectives, build_outlier_weights, calc_correlations_from_weights, \
    drop_metrics_from_weights_by_key
from bounos.ChartBuilders import format_axes, latexify, unique_cm_dict_from_list, plot_axes_views_from_positions_frame
from bounos.ChartBuilders import _texcol, _texcolhalf, _texcolthird, _texfac
from bounos.ChartBuilders.ssp import UUV_time_delay, ssp_function, SSPS

from scripts.publication_scripts import phys_keys, comm_keys, phys_keys_alt, comm_keys_alt, key_order, observer, target, \
    n_metrics, results_path, \
    fig_basedir, subset_renamer, metric_subset_analysis, format_features, savefig

###########
# OPTIONS #
###########
_texcol = 0.5
_texfac = 0.9
use_temp_dir = False
show_outputs = False
recompute = False
img_extn = 'pdf'
_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing

observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9
key_d = {
    'full': key_order,
    'comms': comm_keys,
    'phys': phys_keys,
    'comms_alt': comm_keys_alt,
    'phys_alt': phys_keys_alt
}

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
fig_basedir = "/home/bolster/src/thesis/Figures/"
json_path = "/home/bolster/Dropbox/aietes_json_results"

shared_h5_path = '/dev/shm/shared.h5'
if os.path.exists(shared_h5_path):
    try:
        with pd.get_store(shared_h5_path) as store:
            store.get('joined_target_weights' + "_signed")
    except (AttributeError, KeyError):
        print("Forcing recompute as the thing I expected to be there, well, isn't")
        recompute = True
else:
    print("Forcing recompute as the thing I expected to be there, well, doesn't exist")
    recompute = True

assert os.path.isdir(fig_basedir)


from bounos.Analyses.Dixon import  *



class smart_dict(dict):
    # Replaces missing lookups with the key
    def __missing__(self, key):
        return key

def plot_behaviour_metric_graph(data, fig_path, observer = 'Bravo', target='Alfa', run = 0,
                                force_ymin=None, title=None, y_label_map=None,
                                save_fig=False, show_title=False, suffix=""):

    _bevs = set(data.index.get_level_values('var'))
    _metrics = list(data.columns)
    fig = plt.figure(figsize=(8, 5), dpi=80)
    base_ax = fig.add_axes([0, 0, 1, 1], )
    base_ax.set_axis_off()
    gs = GridSpec(len(_metrics), len(_bevs))
    axes = [[None for _ in range(len(_metrics))] for _ in range(len(_bevs))]
    lines = []

    # Column Behaviours
    for i, (behaviour, bev_df) in enumerate(data.groupby(level='var')):
        # Highlight Target on misbehaviours, if no misbehaviour, highlight all
        highlight_target = defaultdict(lambda: False)
        highlight_target[target] = True if behaviour != 'Waypoint' else False
        # Row Metrics
        metrics_df = bev_df
        if 'run' in bev_df.index.names:
            metrics_df = metrics_df.xs(run, level='run', drop_level=True)
        if 'observer' in bev_df.index.names:
            metrics_df = metrics_df.xs(observer, level='observer')
        for j, metric in enumerate(metrics_df):
            ax = fig.add_subplot(gs[j, i])
            for target_name, target_df in metrics_df[metric].groupby(level='target'):
                index = target_df.index.get_level_values('t') # in Seconds
                index = [ I/60.0 for I in index]# in minutes
                line = ax.plot(index, target_df.values,
                        label=target_name,
                        lw=1,
                        alpha=1.0 if highlight_target[target_name] else 0.8)

                # First Metric Behaviour (Legend)
                if j == 0:
                    if i == 0:
                        lines.append(line[0])
                    else:
                        pass

                loc = mtick.MaxNLocator(nbins=7, integer=True)  # this locator puts ticks at regular intervals
                ax.xaxis.set_major_locator(loc)
            if j == 0:
                ax.set_title(behaviour)
            ax.grid(True, alpha=0.2)
            ax.autoscale_view(scalex=False, tight=True)

            # Metric label on left most graph
            if i == 0:
                if y_label_map is None:
                    ax.set_ylabel(metric)
                else:
                    ax.set_ylabel(smart_dict(y_label_map)[metric])

            # Last Metric Behaviour (Xlabel)
            if j == len(_metrics) - 1:
                ax.set_xlabel("Mission Time (min)")
            else:
                [l.set_visible(False) for l in ax.get_xticklabels()]
            ax.grid(color='lightgray')
            axes[i][j] = ax

    # For each metric row
    for j in range(len(_metrics)):
        (m_ymin, m_ymax) = (float('inf'), float('-inf'))
        (m_xmin, m_xmax) = (float('inf'), float('-inf'))
        # Take the max limits across all behaviours
        for i in range(len(_bevs)):
            (ymin, ymax) = axes[i][j].get_ylim()
            (xmin, xmax) = axes[i][j].get_xlim()
            m_ymax = max(ymax, m_ymax)
            m_ymin = min(ymin, m_ymin)
            m_xmax = max(xmax, m_xmax)
            m_xmin = min(xmin, m_xmin)

        m_ymin = m_ymin if force_ymin is None else force_ymin
        # Reset the rows limits
        for i in range(len(_bevs)):
            axes[i][j].set_ylim((m_ymin*0.9, m_ymax * 1.1))
            axes[i][j].set_xlim((0, m_xmax))

    fig.legend(lines, dict_levels(bev_df)['target'].tolist(), 'center right',
               bbox_to_anchor=(1.0,0.5), ncol=1, labelspacing=2)
    fig.subplots_adjust(
            left=0.05, bottom=0.13, right=0.9, top=0.96, wspace=0.2, hspace=0.05)
    if title is not None and show_title:
        fig.suptitle(title, fontsize=12)
    if save_fig:
        savefig(fig, os.path.join(fig_path,title.replace(' ','_')+suffix), transparent=True, tight=False)
    else:
        fig.show()

    return fig

class SingleRunGraphing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(SingleRunGraphing, cls).setUpClass()
        #Grab results h5 file from set "good" results
        cls.fig_path = fig_basedir
        results_path_good_as_of_monday210416 = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2016-03-22-01-56-00"
        results_path_multi_run = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2016-03-24-19-38-14"
        cls.results_path = results_path_multi_run

        cls.run_id = 0
        cls.target = 'Alfa'
        cls.observer = 'Bravo'
        cls.trust_period = 10

        with pd.get_store(cls.results_path + '.h5') as store:
            cls.positions = store.positions
            cls.trust_observations = store.trust.drop('Tail',level='var')
            cls.trust_observations.columns.name = 'metric'
            map_levels(cls.trust_observations, var_rename_dict)
            # Convert Sample Times into proper stuff
            cls.trust_observations.reset_index(inplace=True)
            cls.trust_observations['t']*=cls.trust_period
            cls.trust_observations.set_index(['var','run','observer','t','target'], inplace=True)

        data = cls.trust_observations.xs(cls.observer, level='observer').xs(cls.run_id, level='run')

        cls.deviance = pd.concat([gf - gf.mean() for g, gf in data.groupby(level=['var', 't'])])
        cls.sigmas = pd.concat([(gf / (gf.std(axis=0))).abs() for g, gf in cls.deviance.groupby(level=['var', 't'])])

    def test_position_track_plotting(self):
        fig = plot_axes_views_from_positions_frame(self.positions.xs('Waypoint', level='var').xs(self.run_id,level='run'))
        savefig(fig,os.path.join(self.fig_path, "Waypoint_Position_Track_Run_{0:d}".format(self.run_id)),
                    transparent=True)

    def test_raw_metric_value_graph(self):
        fig = plot_behaviour_metric_graph(self.trust_observations, fig_path=self.fig_path, observer=self.observer,
                                          run=self.run_id, y_label_map={'Speed': 'Speed ($ms^{-1})$'},
                                          title="Metric Values", save_fig=True)
    def test_first_order_deviance_graph(self):
        fig = plot_behaviour_metric_graph(self.deviance, fig_path=self.fig_path, observer=self.observer,
                                          run=self.run_id, y_label_map={'Speed': 'Speed ($ms^{-1})$'},
                                          title="Metric Deviation", save_fig=True)
    def test_sigma_deviance_graph(self):
        fig = plot_behaviour_metric_graph(self.sigmas, fig_path=self.fig_path, observer=self.observer,
                                          run=self.run_id, force_ymin=1.0,
                                          y_label_map={'INDD': 'INDD ($\sigma$)',
                                                       'INHD': 'INHD ($\sigma$)',
                                                       'Speed': 'Speed ($\sigma$)'},
                                          title="Metric Sigma Deviance", save_fig=True)
    def test_per_metric_sigma_bars(self):
        offset = self.sigmas.index.get_level_values('t').values.max() / self.trust_period
        latexify(columns=_texcolhalf)
        fig_all, axes_all = plt.subplots(nrows=1, ncols=3,figsize=(6, 2.2), dpi=80)
        for i, metric in enumerate(self.sigmas.columns):
            summed_sigmas = self.sigmas.unstack('target')[metric].groupby(level='var').sum()

            summed_sigmas['Mean'] = summed_sigmas.mean(axis=1)
            _ = (summed_sigmas / offset-1).plot(ax=axes_all[i], kind='bar', legend=False)
            axes_all[i].grid(color='lightgray')
            axes_all[i].set_xlabel(metric)
            axes_all[i].set_ylabel('Overall $\sigma$', labelpad=-5)
            for tick in axes_all[i].get_xticklabels():
                tick.set_rotation(0)
            axes_all[i].set_ylim([-1.0,1.0])

        axes_all = format_axes(axes_all)

        lines, labels = axes_all[-1].get_legend_handles_labels()

        leg=fig_all.legend(lines, labels, 'center right',
                       bbox_to_anchor=(1.0, 0.5), ncol=1,
                       labelspacing=1, handleheight=0.2)


        fig_all.subplots_adjust(
            left=0.051, bottom=0.15, right=0.88, top=0.978, wspace=0.2, hspace=0.05)
        savefig(fig_all, os.path.join(self.fig_path, "summedsigmabar"), tight=False, transparent=True)

        for i, metric in enumerate(self.sigmas.columns):
            summed_sigmas = self.sigmas.unstack('target')[metric].groupby(level='var').sum()
            fig,ax = plt.subplots(1,1,figsize=latexify(columns=_texcolthird, factor=1.2))
            summed_sigmas['Mean'] = summed_sigmas.mean(axis=1)
            _ = (summed_sigmas / offset-1).plot(ax=ax, kind='bar', legend=False)
            ax.set_xlabel('Behaviour')
            ax.set_ylabel('Overall $\sigma$')
            for tick in ax.get_xticklabels():
                tick.set_rotation(0)
            ax = format_axes(ax)
            ax.set_ylim([-1.0,1.0])
            if metric == self.sigmas.columns[-1]:
                ax.legend(ncol=2)
            savefig(fig, os.path.join(self.fig_path, "summedsigmabar_{0}".format(metric)), transparent=True)

def group_dixon_test(g,gf):
    acc = []
    for k,v in gf.T.apply(dixon_test_95, axis=1).T.iteritems():
        outlier = map(lambda x:x[-1],gf.loc[gf[k]==v].index.tolist())
        if outlier:
            acc.append((g[0],g[1],k,outlier[0]))
        else:
            acc.append((g[0],g[1],k,'None'))

        if len(outlier)>1:
            raise ValueError("Haven't written the case for multiple outliers yet")
    return pd.DataFrame(acc, columns=['var','run','metric','target'])

class MultiRunSigmaAndConfidenceGraphing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(MultiRunSigmaAndConfidenceGraphing, cls).setUpClass()
        #Grab results h5 file from set "good" results
        results_path_multi_run = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2016-03-24-21-10-00"
        cls.results_path = results_path_multi_run
        cls.fig_path = fig_basedir

        cls.target = 'Alfa'
        cls.observer = 'Bravo'
        cls.good_bev = 'Control'
        cls.trust_period = 10

        with pd.get_store(cls.results_path + '.h5') as store:
            cls.trust_observations = store.trust.drop('Tail',level='var')
            cls.trust_observations.columns.name = 'metric'
            map_levels(cls.trust_observations, {'Waypoint': cls.good_bev})
            # Convert Sample Times into proper stuff
            cls.trust_observations.reset_index(inplace=True)
            cls.trust_observations['t']*=cls.trust_period
            cls.trust_observations.set_index(['var','run','observer','t','target'], inplace=True)

        data = cls.trust_observations.xs(cls.observer, level='observer')
        cls.deviance = pd.concat([gf - gf.mean() for g, gf in data.groupby(level=['var', 'run', 't'])])
        cls.sigmas = pd.concat([(gf / (gf.std(axis=0))).abs() for g, gf in cls.deviance.groupby(level=['var', 'run', 't'])])
        # Optimise the below; it's naaaaasty for long runs
        cls.summed_sigmas = cls.sigmas.unstack('target').groupby(level=['var','run']).sum()
        cls.dixon_df = pd.concat(
            [(group_dixon_test(g, gf)) for g, gf in cls.summed_sigmas.stack('target').groupby(level=['var', 'run'])])
        cls.dixon_df['correct'] = (cls.dixon_df['target'] == cls.target) != ((cls.dixon_df['var'] == cls.good_bev) & (cls.dixon_df['target'] == 'None'))

    def test_overall_stats(self):
        overall_stats = self.dixon_df.groupby('var').describe().unstack('var').correct.iloc[[1, 2]].T
        overall_stats.columns = ['Mean', 'Std']
        overall_stats.index.name = 'Behaviour'
        with open(os.path.join(self.fig_path,"overall_stats.tex"), 'w') as f:
            overall_stats.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_per_metric_stats(self):
        per_metric_stats = self.dixon_df.groupby(['var', 'metric']).describe().unstack(['var', 'metric']).iloc[[1, 2]].stack(
            'var').correct
        per_metric_stats.columns.name = 'Metric'
        per_metric_stats.index.names = ['', 'Behaviour']
        with open(os.path.join(self.fig_path,"per_metric_stats.tex"), 'w') as f:
            per_metric_stats.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_summed_sigma_confidence_table(self):
        intergfs = []
        for g, gf in self.summed_sigmas.stack('target').groupby(level=['var','run']):
            _intergf = gf.unstack('target').stack('metric')
            intergfs.append(_intergf[self.target] / _intergf.drop(self.target, axis=1).mean(axis=1))

        confidences = pd.concat(intergfs).unstack('metric')
        c_table = confidences.groupby(level='var').describe().unstack('var').iloc[[1, 2]].stack('var')
        c_table.columns.name = 'Metric'
        c_table.index.names = ['', 'Behaviour']
        with open(os.path.join(self.fig_path,"confidence.tex"), 'w') as f:
            c_table.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_classifer_minority_test(self):

        def _classifier(gf):
            return classifier_phys(gf, minority=True, _dixon_test=dixon_test_95)

        df = pd.DataFrame(self.summed_sigmas.stack('target').groupby(level=['var', 'run']).apply(_classifier))
        df[['suspect', 'suspicion']] = df[0].apply(pd.Series)
        df.reset_index(inplace=True)
        df = df[['var', 'suspect', 'suspicion']]
        correct = lambda r: r['var'] == r['suspicion']
        df['correct'] = df.apply(correct, axis=1)
        df=df.groupby('var').mean()
        df.columns = ['Probability of Correct Blind Identification']
        df.index.names = ['True Behaviour']
        with open(os.path.join(self.fig_path, "classifier_minority.tex"), 'w') as f:
            df.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_classifer_test(self):

        def _classifier(gf):
            return classifier_phys(gf, minority=False, _dixon_test=dixon_test_95)

        df = pd.DataFrame(self.summed_sigmas.stack('target').groupby(level=['var', 'run']).apply(_classifier))
        df[['suspect', 'suspicion']] = df[0].apply(pd.Series)
        df.reset_index(inplace=True)
        df = df[['var', 'suspect', 'suspicion']]
        correct = lambda r: r['var'] == r['suspicion']
        df['correct'] = df.apply(correct, axis=1)
        df=df.groupby('var').mean()
        df.columns = ['Probability of Correct Blind Identification']
        df.index.names = ['True Behaviour']
        with open(os.path.join(self.fig_path, "classifier.tex"), 'w') as f:
            df.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

def classifier_phys(gf, minority=False, _dixon_test=dixon_test_95):
    # Find ANY qtest activations
    qtest = gf.T.apply(_dixon_test, axis=1)
    n_outliers = len(filter(lambda l: l != [None], qtest))

    if n_outliers > minority:
        # Find the outlier
        outliers = filter(lambda l: l != [],
                          [map(lambda x: x[-1],
                               gf.loc[gf[k] == v].index.tolist())
                           for k, v in qtest.T.iteritems()]
                          )
        outlier = set([i[0] for i in outliers])
        if len(outlier) > 1:
            print("Sneaky Contradictions!{}".format(outlier))
        outlier = outlier.pop()

        # Apply this to groups of var/run of summed_sigma
        _intergf = gf.unstack('target').stack('metric')
        confidence = (_intergf[outlier] / _intergf.drop(outlier, axis=1).mean(axis=1)).unstack('metric')

        if confidence['Speed'].values > 1.75:
            return (outlier, "Shadow")
        else:
            return (outlier, "SlowCoach")
    else:
        return (None, "Control")

class MultiRunSigmaAndConfidenceGraphing_Multi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(MultiRunSigmaAndConfidenceGraphing_Multi, cls).setUpClass()
        #Grab results h5 file from set "good" results
        cls.results_path = results_path
        cls.fig_path = fig_basedir

        cls.target = 'Alfa'
        cls.observer = 'Bravo'
        cls.good_bev = 'Fair'
        cls.trust_period = 10

        with pd.get_store(cls.results_path + '.h5') as store:
            cls.trust_observations.columns.name = 'metric'
            map_levels(cls.trust_observations, {'Waypoint': cls.good_bev})
            # Convert Sample Times into proper stuff
            cls.trust_observations.reset_index(inplace=True)
            cls.trust_observations['t']*=cls.trust_period
            cls.trust_observations.set_index(['var','run','observer','t','target'], inplace=True)

        data = cls.trust_observations.xs(cls.observer, level='observer')
        cls.deviance = pd.concat([gf - gf.mean() for g, gf in data.groupby(level=['var', 'run', 't'])])
        cls.sigmas = pd.concat([(gf / (gf.std(axis=0))).abs() for g, gf in cls.deviance.groupby(level=['var', 'run', 't'])])
        # Optimise the below; it's naaaaasty for long runs
        cls.summed_sigmas = cls.sigmas.unstack('target').groupby(level=['var','run']).sum()
        cls.dixon_df = pd.concat(
            [(group_dixon_test(g, gf)) for g, gf in cls.summed_sigmas.stack('target').groupby(level=['var', 'run'])])
        cls.dixon_df['correct'] = (cls.dixon_df['target'] == cls.target) != ((cls.dixon_df['var'] == cls.good_bev) & (cls.dixon_df['target'] == 'None'))

    def test_overall_stats(self):
        overall_stats = self.dixon_df.groupby('var').describe().unstack('var').correct.iloc[[1, 2]].T
        overall_stats.columns = ['Mean', 'Std']
        overall_stats.index.name = 'Behaviour'
        with open(os.path.join(self.fig_path,"overall_stats_multi.tex"), 'w') as f:
            overall_stats.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_per_metric_stats(self):
        per_metric_stats = self.dixon_df.groupby(['var', 'metric']).describe().unstack(['var', 'metric']).iloc[[1, 2]].stack(
            'var').correct
        per_metric_stats.columns.name = 'Metric'
        per_metric_stats.index.names = ['', 'Behaviour']
        with open(os.path.join(self.fig_path,"per_metric_stats_multi.tex"), 'w') as f:
            per_metric_stats.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_summed_sigma_confidence_table(self):
        intergfs = []
        for g, gf in self.summed_sigmas.stack('target').groupby(level=['var','run']):
            _intergf = gf.unstack('target').stack('metric')
            intergfs.append(_intergf[self.target] / _intergf.drop(self.target, axis=1).mean(axis=1))

        confidences = pd.concat(intergfs).unstack('metric')
        c_table = confidences.groupby(level='var').describe().unstack('var').iloc[[1, 2]].stack('var')
        c_table.columns.name = 'Metric'
        c_table.index.names = ['', 'Behaviour']
        with open(os.path.join(self.fig_path,"confidence_multi.tex"), 'w') as f:
            c_table.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_classifer_minority_test(self):

        def _classifier(gf):
            return classifier_multi(gf, minority=True, _dixon_test=dixon_test_95)

        df = pd.DataFrame(self.summed_sigmas.stack('target').groupby(level=['var', 'run']).apply(_classifier))
        df[['suspect', 'suspicion']] = df[0].apply(pd.Series)
        df.reset_index(inplace=True)
        df = df[['var', 'suspect', 'suspicion']]
        correct = lambda r: r['var'] == r['suspicion']
        df['correct'] = df.apply(correct, axis=1)
        df=df.groupby('var').mean()
        df.columns = ['Probability of Correct Blind Identification']
        df.index.names = ['True Behaviour']
        with open(os.path.join(self.fig_path, "classifier_minority_multi.tex"), 'w') as f:
            df.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

    def test_classifer_test(self):

        def _classifier(gf):
            return classifier_multi(gf, minority=False, _dixon_test=dixon_test_95)

        df = pd.DataFrame(self.summed_sigmas.stack('target').groupby(level=['var', 'run']).apply(_classifier))
        df[['suspect', 'suspicion']] = df[0].apply(pd.Series)
        df.reset_index(inplace=True)
        df = df[['var', 'suspect', 'suspicion']]
        correct = lambda r: r['var'] == r['suspicion']
        df['correct'] = df.apply(correct, axis=1)
        df=df.groupby('var').mean()
        df.columns = ['Probability of Correct Blind Identification']
        df.index.names = ['True Behaviour']
        with open(os.path.join(self.fig_path, "classifier_multi.tex"), 'w') as f:
            df.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))


def classifier_multi(gf, minority=False, _dixon_test=dixon_test_95):
    # Find ANY qtest activations
    qtest = gf.T.apply(_dixon_test, axis=1)
    n_outliers = len(filter(lambda l: l != [None], qtest))

    if n_outliers > minority:
        # Find the outlier
        outliers = filter(lambda l: l != [],
                          [map(lambda x: x[-1],
                               gf.loc[gf[k] == v].index.tolist())
                           for k, v in qtest.T.iteritems()]
                          )
        outlier = set([i[0] for i in outliers])
        if len(outlier) > 1:
            print("Sneaky Contradictions!{}".format(outlier))
        outlier = outlier.pop()

        # Apply this to groups of var/run of summed_sigma
        _intergf = gf.unstack('target').stack('metric')
        confidence = (_intergf[outlier] / _intergf.drop(outlier, axis=1).mean(axis=1)).unstack('metric')

        if confidence['Speed'].values > 1.75:
            return (outlier, "Shadow")
        else:
            return (outlier, "SlowCoach")
    else:
        return (None, "TEST")

class ThesisOneShotDiagrams(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def testSSPPlots(self):
        depth = 100
        for ssp in SSPS:
            figsize=latexify(columns=0.5)
            r=UUV_time_delay(SSP=ssp, graph=1, depth=depth, dist_calc=False,
                             pdf_plot={'filepath':os.path.join(fig_basedir,"ssp_{}".format(ssp)),
                                       'figsize':figsize})
    def testTempSalGlobes(self):
        resolution = '0.25'
        def make_map(cube, projection=ccrs.InterruptedGoodeHomolosine(), figsize=None,
                     cmap=cm.avhrr, label='temperature'):
            fig, ax = plt.subplots(figsize=figsize,
                                   subplot_kw=dict(projection=projection))
            ax.add_feature(cfeature.LAND, facecolor='0.75')
            cs = iplt.pcolormesh(cube, cmap=cmap)
            ax.coastlines()
            if isinstance(projection, ccrs.PlateCarree) or isinstance(projection, ccrs.Mercator):
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5,
                                  color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = gl.ylabels_right = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
            cbar = dict(extend='both', shrink=0.5, pad=0.02,
                        orientation='horizontal', fraction=0.1)
            cb = fig.colorbar(cs, **cbar)
            if label == 'temperature':
                cb.ax.set_xlabel(r"$^{\circ}$ C")
            elif label == 'salinity':
                cb.ax.set_xlabel(r"Salinity $(ppt)$")
            else:
                cb.ax.set_xlabel(label)
            return fig, ax

        bbox = [2.5, 357.5, -87.5, 87.5]
        # Temperature
        kw = dict(bbox=bbox, variable='temperature', clim_type='00',
                  resolution=resolution, full=True)#
        cubes = woa_subset(**kw)
        cube = cubes[4]
        c = cube[0, 0, ...]
        fig, ax = make_map(c, figsize=latexify())
        savefig(fig, os.path.join(fig_basedir, 'temp_globe'), bbox_inches='tight')
        # Salinity
        kw = dict(bbox=bbox, variable='salinity', clim_type='00',
                  resolution=resolution, full=True)#
        cubes = woa_subset(**kw)
        cube = cubes[5]
        c = cube[0, 0, ...]
        fig, ax = make_map(c, figsize=latexify(), label='salinity')
        savefig(fig, os.path.join(fig_basedir, 'sal_globe'), bbox_inches='tight')

    def testTempSalProfiles(self):
        def _cube_decomposer(cube):
            return cube[0, :].data, cube.coord(axis='Z').points

        def plot_profile(ax,cube, label):
            x,z = _cube_decomposer(cube)
            l = ax.plot(x, z, label=label, linewidth=2)

        def ssp_decuber(S, T):
            s, zs = _cube_decomposer(S)
            t, zt = _cube_decomposer(T)
            return ssp(s, t, zs)

        def ssp(t,s,d):
            v=1448.96+\
              4.591*t-\
              (5.304*10**-2)*(np.power(t,2))+\
              (2.374*10**-4)*(np.power(t,3))+\
              (1.34*(s-35))+\
              (1.63*10**-2)*d+\
              (1.675*10**-7)*(np.power(d,2))-\
              (1.025*10**-2)*(t*(s-25))-\
              (7.139*10**-13)*(t*np.power(d,3))
            return v

        fig, (axT, axS, axV) = plt.subplots(1,3,sharey=True,figsize=(5, 6))
        kw = dict(variable='temperature', clim_type='00', resolution='1.00', full=False)
        polar_t = woa_profile(-25.5, -70.5, **kw)
        tempe_t = woa_profile(-25.5, -50.0, **kw)
        equat_t = woa_profile(-25.5, 0, **kw)
        plot_profile(axT,polar_t, label='Polar')
        plot_profile(axT,tempe_t, label='Temperate')
        plot_profile(axT,equat_t, label='Equatorial')
        axT.invert_yaxis()
        axT.set_xlabel(r"$^{\circ}$ C")
        axT.set_ylabel(r"Depth $(m)$")
        _ = axT.set_ylim(200,0)


        kw = dict(variable='salinity', clim_type='00', resolution='1.00', full=False)
        polar_s = woa_profile(-25.5, -70.5, **kw)
        tempe_s = woa_profile(-25.5, -50.0, **kw)
        equat_s = woa_profile(-25.5, 0, **kw)
        plot_profile(axS,polar_s, label='Polar')
        plot_profile(axS,tempe_s, label='Temperate')
        plot_profile(axS,equat_s, label='Equatorial')
        axS.invert_yaxis()
        axS.set_xlabel(r"Salinity $(ppt)$")
        _ = axS.set_ylim(200,0)
        axS.xaxis.set_major_locator(mtick.MaxNLocator(5))

        _, z = _cube_decomposer(polar_s)
        polar_v = ssp_decuber(polar_s, polar_t)
        tempe_v = ssp_decuber(tempe_s, tempe_t)
        equat_v = ssp_decuber(equat_s, equat_t)
        axV.plot(polar_v, z, label='Polar', linewidth=2)
        axV.plot(tempe_v, z, label='Temperate', linewidth=2)
        axV.plot(equat_v, z, label='Equatorial', linewidth=2)
        axV.invert_yaxis()
        axV.set_xlabel(r"SSP $(ms^{-1})$")
        _ = axV.set_ylim(200, 0)
        _ = axV.set_xlim(1500, 1550)
        axV.xaxis.set_major_locator(mtick.MaxNLocator(3))
        lines, labels = axV.get_legend_handles_labels()

        fig.subplots_adjust(
            left=0.05, bottom=0.06, right=0.98, top=0.965, wspace=0.2, hspace=0.05)

        fig.legend(lines, labels,'upper center',
                   bbox_to_anchor=(0.5, 1), ncol=3)
        savefig(fig, os.path.join(fig_basedir, 'temp_sal_profile'), tight=False)


    def testThreatSurfacePlot(self):
        fig_filename = os.path.join(fig_basedir,'threat_surface_sum')
        fig_size = latexify(columns=_texcol, factor=_texfac)
        fig_size = (fig_size[0], fig_size[1] / 2)

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=fig_size,
                                 sharex='none', sharey='none',
                                 subplot_kw={'axisbg': (1, 0.9, 0.9), 'alpha': 0.1})
        for ax in axes:
            ax.set_aspect('equal')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ys = np.linspace(0.15, 0.85, 4)

        # Single
        ax = axes[0]
        ax.set_title("Single Metric")
        ax.add_patch(plt.Circle((0.1, ys[0]), radius=0.075, color='g', alpha=0.9))
        ax.annotate('single metric\nobservation', xy=(0.15, 0.2), xycoords='data',
                    xytext=(-5, 25), textcoords='offset points',

                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc,angleA=0180,armA=30,rad=10"),
                    )

        # Vector
        ax = axes[1]
        ax.set_title("Multi Metric")
        ax.add_patch(plt.Rectangle((0.0, ys[0] - 0.1), 0.99, 0.2, alpha=0.2))
        for x in np.linspace(0.1, 0.9, 5):
            ax.add_patch(plt.Circle((x, ys[0]), radius=0.075, color='g', alpha=0.9))
        ax.annotate('single domain\nobservation', xy=(0.5, 0.3), xycoords='data',
                    xytext=(-30, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc,angleA=0180,armA=30,rad=10"),
                    )

        # Multi
        ax = axes[2]
        ax.set_title("Multi Domain")
        ax.text(1, 0.3, r"$\}$", fontsize=52)
        ax.text(1.3, 0.42, "Combination of\nmultiple vectors,\neach containing\nmultiple metrics",
                verticalalignment='center')

        ax.add_patch(plt.Rectangle((0.0, ys[0] - 0.1), 0.99, 0.2, alpha=0.2))
        for x in np.linspace(0.1, 0.9, 5):
            ax.add_patch(plt.Circle((x, ys[0]), radius=0.075, color='g', alpha=0.9))

        ax.add_patch(plt.Rectangle((0.0, ys[1] - 0.1), 0.99, 0.2, alpha=0.2))
        for x in np.linspace(0.2, 0.8, 3):
            ax.add_patch(plt.Circle((x, ys[1]), radius=0.075, color='y', alpha=0.9))

        ax.add_patch(plt.Rectangle((0.0, ys[2] - 0.1), 1.0, 0.2, alpha=0.2))
        for x in np.linspace(0.1, 0.9, 6):
            ax.add_patch(plt.Circle((x, ys[2]), radius=0.075, color='b', alpha=0.9))

        ax.add_patch(plt.Rectangle((0.0, ys[3] - 0.1), 1.0, 0.2, alpha=0.2))
        for x in np.linspace(0, 1, 10):
            ax.add_patch(plt.Circle((x, ys[3]), radius=0.075, color='r', alpha=0.9))

        axes = map(format_axes, axes)
        fig.delaxes(axes[3])
        savefig(fig, fig_filename, transparent=False)

        self.assertTrue(os.path.isfile(fig_filename + '.' + img_extn))

        if show_outputs:
            try_to_open(fig_filename + '.' + img_extn)

class ThesisDiagrams(unittest.TestCase):
    signed = True
    runtime_computed = False #CHANGE ME BACK
    complete_subsets = False

    @classmethod
    def setUpClass(self):
        if use_temp_dir:
            self.dirpath = tempfile.mkdtemp()
        else:
            self.dirpath = fig_basedir

        os.chdir(self.dirpath)

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
        if recompute or self.runtime_computed:
            self.recompute_features_in_shared(signed=self.signed)
            self.runtime_computed = True
        else:
            print("Apparently we don't need to recompute! Great!")

        dumping_suffix = "_signed" if self.signed else "_unsigned"
        with pd.get_store(shared_h5_path) as store:
            self.joined_target_weights = store.get('joined_target_weights' + dumping_suffix)
            self.joined_feats = store.get('joined_feats' + dumping_suffix)
            self.comms_only_weights = store.get('comms_only_weights' + dumping_suffix)
            self.comms_only_feats = store.get('comms_only_feats' + dumping_suffix)
            self.phys_only_weights = store.get('phys_only_weights' + dumping_suffix)
            self.phys_only_feats = store.get('phys_only_feats' + dumping_suffix)

            self.comms_alt_only_weights = store.get('comms_alt_only_weights' + dumping_suffix)
            self.comms_alt_only_feats = store.get('comms_alt_only_feats' + dumping_suffix)
            self.phys_alt_only_weights = store.get('phys_alt_only_weights' + dumping_suffix)
            self.phys_alt_only_feats = store.get('phys_alt_only_feats' + dumping_suffix)

        self.joined_feat_weights = categorise_dataframe(non_zero_rows(self.joined_feats).T)
        self.comms_feat_weights = categorise_dataframe(non_zero_rows(self.comms_only_feats).T)
        self.phys_feat_weights = categorise_dataframe(non_zero_rows(self.phys_only_feats).T)

        self.comms_alt_feat_weights = categorise_dataframe(non_zero_rows(self.comms_alt_only_feats).T)
        self.phys_alt_feat_weights = categorise_dataframe(non_zero_rows(self.phys_alt_only_feats).T)

        # Consistent Colouring for Metrics (Currently only applies to relevance charts)
        self.metric_colour_map = unique_cm_dict_from_list(self.joined_feats.keys().tolist())
        # Also need to handle latexified keys but use the same colour:
        for k, v in self.metric_colour_map.items():
            self.metric_colour_map[metric_rename_dict[k]] = v
        # Same for nodes
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()
            v = list(trust_observations.index.levels[2])
            if hasattr(cmap_qual, "Dark2_{}".format(len(v))):
                _cmap = getattr(cmap_qual, "Dark2_{}".format(len(v))).mpl_colormap
            else:
                _cmap = 'nipy_spectral'
            self.node_colour_map = unique_cm_dict_from_list(v, cmap=_cmap)

    @classmethod
    def recompute_features_in_shared(cls, signed=False):
        dumping_suffix = "_signed" if signed else "_unsigned"
        # All Metrics
        print("Building Joined {0} Target Weights".format("Signed" if signed else "Unsigned"))
        joined_target_weights = build_outlier_weights(results_path + "/outliers.bkup.h5",
                                                      observer=observer, target=target,
                                                      n_metrics=n_metrics, signed=signed)
        if cls.complete_subsets:
            print("Generating Complete Metric Powerset assessments, this might take a while")
            cls.compute_complete_metric_subsets(joined_target_weights, dumping_suffix)
            #TODO Automatically remap the complete and subsets below to take from the above if we're doing big work
        joined_feats = format_features(
            target_weight_feature_extractor(
                joined_target_weights
            )
        )[key_order]
        print("Dumping Joined Target Weights")
        joined_target_weights.to_hdf(shared_h5_path, 'joined_target_weights' + dumping_suffix)
        joined_feats.to_hdf(shared_h5_path, 'joined_feats' + dumping_suffix)


        _, _ = cls.metric_subset_weight_and_feature_extractor('comms', comm_keys, joined_target_weights, dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('phys', phys_keys, joined_target_weights, dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('comms_alt', comm_keys_alt, joined_target_weights,
                                                              dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('phys_alt', phys_keys_alt, joined_target_weights,
                                                              dumping_suffix)

    @classmethod
    def metric_subset_weight_and_feature_extractor(cls, subset_str, desired_keys, complete_target_weights,
                                                   dumping_suffix, restart=True):
        """

        :param subset_str: Set of string metric indices to power-set over
        :param desired_keys: argument to metric key inversion
        :param complete_target_weights:
        :param dumping_suffix:
        :param restart: attempt to recover from a broken restart by inspecting the shared store for this subset
        :return:
        """
        subset_only_weights = None
        subset_only_feats = None

        dump_weight_str = "{0}_only_weights{1}".format(subset_str, dumping_suffix)
        dump_feat_str = "{0}_only_feats{1}".format(subset_str, dumping_suffix)

        if restart:
            with pd.get_store(shared_h5_path) as s:
                if dump_weight_str in s and dump_feat_str in s:
                    print("Recalling {0} Weight and Feats from shared store".format(subset_str))
                    subset_only_weights = s.get(dump_weight_str)
                    subset_only_feats = s.get(dump_feat_str)

        if subset_only_weights is None or subset_only_feats is None:
            print("Building {0} Target Weights".format(subset_str))
            subset_only_weights = drop_metrics_from_weights_by_key(
                complete_target_weights,
                metric_key_inverter(desired_keys)
            )
            try:
                subset_only_feats = format_features(
                    target_weight_feature_extractor(
                        subset_only_weights
                    )
                )
                print("Dumping {0} Target Weights".format(subset_str))
                subset_only_weights.to_hdf(shared_h5_path, "{0}_only_weights{1}".format(subset_str, dumping_suffix))
                subset_only_feats.to_hdf(shared_h5_path, "{0}_only_feats{1}".format(subset_str, dumping_suffix))
            except ValueError:
                # TODO this is a "monkeypatch"
                #ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.
                print("Subset appears to have no interesting features")

        return subset_only_weights, subset_only_feats

    @classmethod
    def compute_complete_metric_subsets(cls, complete_target_weights, dumping_suffix, min_subset=4):
        """
        Generate Every Optimisation Combination of Metrics with a minimum subset length of min_subset
        #TODO this is lazily parallelisable
        :param complete_target_weights:
        :param dumping_suffix:
        :param min_subset:
        :return:
        """
        for i,subset in tqdm(enumerate(powerset(complete_target_weights.index.names))):
            if len(subset)>min_subset:
                subset_str = '_'.join(subset)
                print("S:{},".format(i), end="")
                _, _ = cls.metric_subset_weight_and_feature_extractor(subset_str, subset, complete_target_weights, dumping_suffix)

    def save_feature_plot(self, feats, fig_filename, hatches=False, annotate=None):
        """
        Plot the Metric Features Extraction plots from Multi-Domain Analysis section


        :param feats:
        :param fig_filename:
        :param hatches: (bool) Print-safe Hatching
        :param annotate: (None, int) Annotate N highest rated significances for each class
        :return:
        """
        fig_size = latexify(columns=_texcol, factor=_texfac)
        # Make sure feats are appropriately ordered for plotting
        sorted_feat_keys = sorted(feats.keys(), key=lambda x: key_order.index(invert_dict(metric_rename_dict)[x]))
        feats = feats[sorted_feat_keys]
        these_feature_colours = [self.metric_colour_map[k] for k in feats.keys().tolist()]

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax = feats[~(feats == 0).all(axis=1)].plot(
            ax=ax, kind='bar', rot=0, width=0.9, figsize=fig_size,
            legend=False, colors=these_feature_colours
        )
        ax.set_xlabel("Behaviour", labelpad=-5)
        ax.set_ylabel("Est. Metric Significance")
        fig = ax.get_figure()
        bars = ax.patches
        if hatches:
            hatches = ''.join(h * 4 for h in ['-', 'x', '\\', '*', 'o', '+', 'O', '.', '_'])
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

        if annotate:
            scenario_bars_d = defaultdict(list)
            for _i, bar in enumerate(bars):
                # patches are added in order, so each 'class' is i/M, and each metric is i%m
                scenario_bars_d[_i // len(these_feature_colours)].append(bar)
            for _i, scenario_bars in scenario_bars_d.items():
                if isinstance(annotate, int):
                    for bar in sorted(scenario_bars, key=lambda b: b.get_height(), reverse=True)[:annotate]:
                        ax.text(bar.get_x() + bar.get_width() / 2, 0.005 + bar.get_height(),
                                "{0:.2}".format(bar.get_height()), ha='center', va='bottom')
                elif isinstance(annotate, float):
                    for bar in filter(lambda b: b.get_height() >= annotate,
                                      sorted(scenario_bars, key=lambda b: b.get_height(), reverse=True)):
                        ax.text(bar.get_x() + bar.get_width() / 2, 0.005 + bar.get_height(),
                                "{0:.2}".format(bar.get_height()), ha='center', va='bottom',fontsize='small')
                else:
                    raise NotImplementedError(
                        "Can't handle an annotation value of type {0}; try float or int".format(type(annotate)))

        ax.legend(loc='best', ncol=1)
        ax = format_axes(ax)
        ax.xaxis.grid(False)  # Disable vertical lines
        savefig(fig, fig_filename, transparent=True)
        self.assertTrue(os.path.isfile(fig_filename + '.' + img_extn))
        if show_outputs:
            try_to_open(fig_filename + '.' + img_extn)



    def testFullMetricTrustRelevance(self):
        fig_filename = 'full_metric_trust_relevance'

        fair_feats = self.joined_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(fair_feats, fig_filename, annotate=0.2)

    def testCommsMetricTrustRelevance(self):
        fig_filename = 'comms_metric_trust_relevance'

        feats = self.comms_only_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(feats, fig_filename, annotate=0.2)

    def testPhysMetricTrustRelevance(self):
        fig_filename = 'phys_metric_trust_relevance'

        feats = self.phys_only_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(feats, fig_filename, annotate=0.2)

    def testFullMetricCorrs(self):
        input_filename = 'input/full_metric_correlations'
        corrs = calc_correlations_from_weights(self.joined_target_weights)

        with open(input_filename + '.tex', 'w') as f:
            f.write(corrs.loc['Fair'].apply(lambda v: np.round(v, decimals=3)).to_latex(escape=False))
        self.assertTrue(os.path.isfile(input_filename + '.tex'))

    def testCommsMetricCorrs(self):
        input_filename = 'input/comms_metric_correlations'
        corrs = calc_correlations_from_weights(self.comms_only_weights)
        with open(input_filename + '.tex', 'w') as f:
            f.write(corrs.loc['Fair'].apply(lambda v: np.round(v, decimals=3)).to_latex(escape=False))
        self.assertTrue(os.path.isfile(input_filename + '.tex'))

    def testPhysMetricCorrs(self):
        input_filename = 'input/phys_metric_correlations'
        corrs = calc_correlations_from_weights(self.phys_only_weights)
        with open(input_filename + '.tex', 'w') as f:
            f.write(corrs.loc['Fair'].apply(lambda v: np.round(v, decimals=3)).to_latex(escape=False))
        self.assertTrue(os.path.isfile(input_filename + '.tex'))

    def testSignedFeatureValidation(self):

        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.xs('Bravo', level='observer', drop_level=False).dropna()
        corrs = calc_correlations_from_weights(self.joined_target_weights)
        feat_weights = self.joined_feat_weights.rename(index=metric_rename_dict) * corrs.apply(np.sign).T
        trust_perspectives = generate_weighted_trust_perspectives(trust_observations,
                                                                  feat_weights,
                                                                  par=False)

    @unittest.skip("I don't understand what this is supposed to be doing nevermind actually doing anything with it....")
    def testRegressions(self):
        """
        :return:
        """
        from sklearn import linear_model, cross_validation
        import scipy as sp

        weight_df = self.joined_target_weights
        target = 'CombinedBadMouthingPowerControl'
        df = weight_df[target].reset_index()
        data = df.drop(target, axis=1).values
        labels = df[target].values

        etr = ske.ExtraTreesRegressor(n_jobs=4, n_estimators=512)
        rtr = ske.RandomForestRegressor(n_jobs=4, n_estimators=512)
        linr = linear_model.LinearRegression()

        for reg in [etr, rtr, linr]:
            scores = cross_validation.cross_val_score(reg, data, labels, scoring='mean_squared_error', n_jobs=4)
            print(scores, sp.stats.describe(scores))

    def test0ValidationBestPlots(self):
        """
        :return:
        """
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()
        map_levels(trust_observations, var_rename_dict)

        def power_set_map(powerset_string):
            strings = powerset_string.split('_')
            metrics = strings[:-3]
            t = strings[-2]
            signed = strings[-1]
            return ({
                'metrics': metrics,
                'type': t,
                'dataset': powerset_string[1:],
                'signed': signed
            })

        results = {}
        if self.complete_subsets:
            try:
                best_d = {}
                # TODO replace the below sequence with a static fileset as no longer doing massively distributed system
                valid_json_files = filter(
                    lambda f: os.stat(os.path.join(json_path, f)).st_size,
                    os.listdir(json_path)
                )
                for f in valid_json_files:
                    subset_str = f.split('.')[-2]
                    best_d[subset_str]=power_set_map(subset_str)
                    with open(os.path.join(json_path, f), 'rb') as fp:
                        best_d[subset_str]['best'] = json.load(fp)

                _key_d = {}
                _weight_d = {}
                for k,v in best_d.items():
                    _key_d[k.lower()]=v['metrics']
                    _weight_d[k.lower()]=v['best']

            except:
                print("Failed building powerset best dict, falling back to static key_d")
                #_key_d = key_d
                # _weight_d = None
                raise

        else:
            _key_d = key_d
            _weight_d = None

        weights = {}
        perspectives = {}
        time_meaned_plots = {}
        alt_time_meaned_plots = {}
        instantaneous_meaned_plots = {}
        alt_instantaneous_meaned_plots = {}

        results = uncpickle("/home/bolster/src/aietes/results/subset_analysis_raw.pkl")
        if len(results) >= 385 and self.complete_subsets:
            print("Successfully recovered from pickle")
        else:
            with Parallel(n_jobs=-1, verbose=10) as par_ctx:
                results = par_ctx(delayed(metric_subset_analysis)
                                                          (trust_observations, key, subset_str, weights_d=_weight_d,
                                                            plot_internal=True, node_palette=self.node_colour_map,
                                                            texcol=_texcolhalf
                                                           )
                                                          for subset_str, key in _key_d.items()
                                                          )

            if self.complete_subsets:
                mkcpickle("/home/bolster/src/aietes/results/subset_analysis_raw.pkl", results)

        for result in results:
            #time_meaned_plots.update(result['time_meaned_plots'])
            #alt_time_meaned_plots.update(result['alt_time_meaned_plots'])
            #instantaneous_meaned_plots.update(result['instantaneous_meaned_plots'])
            #alt_instantaneous_meaned_plots.update(result['alt_instantaneous_meaned_plots'])
            if result is not None:
                perspectives.update(result['trust_perspectives'])
                weights.update(result['weights'])

        inverted_results = defaultdict(list)
        try:
            for (subset_str, target_str), (result) in perspectives.items():
                inverted_results[subset_str].append((target_str, result))
        except ValueError:
            print(perspectives.items()[0])
            raise

        # Time meaned plots and Result inversion
        for (subset_str, target_str), (_, _, result) in time_meaned_plots.items():
            fig_filename = "best_{0}_run_time_{1}".format(subset_str, target_str)
            self.assertTrue(os.path.isfile(fig_filename + img_extn))
            if show_outputs:
                try_to_open(fig_filename + img_extn)

        # instanteous plots
        for (subset_str, target_str), (_, _, result) in instantaneous_meaned_plots.items():
            fig_filename = "best_{0}_run_instantaneous_{1}".format(subset_str, target_str)
            self.assertTrue(os.path.isfile(fig_filename + img_extn))
            if show_outputs:
                try_to_open(fig_filename + img_extn)

        # ALTERNATE Time meaned plots and Result inversion
        for (subset_str, target_str), (_, _, result) in alt_time_meaned_plots.items():
            fig_filename = "best_{0}_run_alt_time_{1}".format(subset_str, target_str)
            self.assertTrue(os.path.isfile(fig_filename + img_extn))
            if show_outputs:
                try_to_open(fig_filename + img_extn)

        # ALTERNATE instanteous plots
        for (subset_str, target_str), (_, _, result) in alt_instantaneous_meaned_plots.items():
            fig_filename = "best_{0}_run_alt_instantaneous_{1}".format(subset_str, target_str)
            self.assertTrue(os.path.isfile(fig_filename + img_extn))
            if show_outputs:
                try_to_open(fig_filename + img_extn)

        # Dump Best weights from best runs to a table
        w_df = pd.concat([pd.Series(weight, index=_key_d[subset.lower()])
                          for (subset, _), weight in weights.items()],
                         keys=weights.keys(),
                         names=['subset', 'target', 'metric']).unstack(level='metric')

        # TODO Guaranteed to break (or give really weird results) when new subsets added
        if len(w_df.index.levels[0]) == 3:
            subset_reindex_keys = ['full', 'comms', 'phys']
        elif len(w_df.index.levels[0]) == 5:
            subset_reindex_keys = ['full', 'comms', 'phys', 'comms_alt', 'phys_alt']
        else:
            subset_reindex_keys = []
            #raise ValueError("Incorrect number of subsets included; {0}".format(w_df.index.levels[0]))

        if subset_reindex_keys:
            w_df = w_df.reindex(subset_reindex_keys, level='subset')

        w_df = w_df[key_order].rename(columns=metric_rename_dict)
        w_df = w_df.unstack('target').rename(subset_renamer).stack('target')

        w_df.to_hdf(os.path.join(results_path,'w_df.h5'), 'weights')

        tex = w_df.to_latex(float_format=lambda x: "{0:1.3f}".format(x), index=True, escape=False,
                            column_format="|l|l|*{{{0}}}{{c|}}".format(len(key_order))) \
            .replace('nan', '') \
            .split('\n')
        tex[2] = '\multicolumn{2}{|c|}{\diagbox{Domain, Behaviour}{Metric}}' + tex[2].lstrip()[
                                                                               1:]  # get rid of the pesky field separator
        tex.pop(3)  # second dimension header; overridden by the above replacement
        # Add hlines between subsets
        hlines = []
        for i, line in enumerate(tex):
            begin = line.lstrip()
            if begin and not begin.startswith('&') and not begin.startswith('\\'):  # if I'm a subset header line
                hlines.append(i)
        for i, j in enumerate(hlines):
            if i:  # first one already covered by midrule
                tex.insert(i + j - 1, '\midrule')  # add a hline above the i found in prev loop
        tex = '\n'.join(tex)
        with open('input/optimised_weights_{}.tex'.format(len(w_df.index.levels[0])), 'w') as f:
            f.write(tex)

        # mkcpickle("/dev/shm/weights", weights)
        mkcpickle("/dev/shm/inverted_results", inverted_results)
        mkcpickle("/dev/shm/subset_reindex_keys", subset_reindex_keys)

        # Generate performance assessments for each weighted metric subset domain against tested behaviour
        self.true_positive_assessment_table(inverted_results, subset_reindex_keys)

        # Generate performance assessments for each weighted metric subset domain against tested behaviour
        # FALSE POSITIVE ASSESSMENT (dT^-)
        self.false_positive_assessment_table(inverted_results, subset_reindex_keys)

        for subset_str, results in inverted_results.items():
            fig_size = latexify(columns=_texcol, factor=_texfac)
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)

            #        _ = _f(trust.unstack('var')[target].xs(observer, level='observer')).boxplot(ax=ax)

            # results.boxplot(ax=ax)
            fig_filename = "box_{0}_run_{1}".format(subset_str, target_str)
            ax.set_title("Example {0} Metric Weighted Assessment for {1}".format(
                subset_renamer(subset_str),
                target_str
            ))
            format_axes(ax)
            savefig(fig, fig_filename, transparent=True)
            self.assertTrue(os.path.isfile(fig_filename + '.' + img_extn))
            if show_outputs:
                try_to_open(fig_filename + '.' + img_extn)


    @classmethod
    def true_positive_assessment_table(cls, inverted_results, subset_reindex_keys):
        perfd = defaultdict()
        for subset_str, results in inverted_results.items():
            _rd = {k: v for k, v in results}
            _df = pd.concat([v for _, v in _rd.items()], keys=_rd.keys(), names=['bev', 'var', 't'])
            df_mean = _df.groupby(level='bev').agg(np.nanmean)
            # Take the mean of mean trust values from all other nodes and subtract suspicious node
            perfd[subset_str] = df_mean.drop(target, axis=1).apply(np.nanmean, axis=1) - df_mean[target]
        perf_df = pd.concat([v for _, v in perfd.items()], keys=perfd.keys(), names=['subset', 'bev']) \
            .unstack('bev').reindex(subset_reindex_keys)
        perf_df['Avg.'] = perf_df.mean(axis=1)
        perf_df = perf_df.append(pd.Series(perf_df.mean(axis=0), name='Avg.'))
        perf_df = perf_df.rename(subset_renamer)
        tex = perf_df.to_latex(float_format=lambda x: "{0:1.2f}".format(x), index=True, column_format="|l|*{4}{c}|r|") \
            .replace('bev', '\diagbox{Domain}{Behaviour}') \
            .split('\n')
        tex.pop(3)  # second dimension header; overridden by the above replacement
        tex.insert(-4, '\hline')  # Hline before averages
        tex = '\n'.join(tex)
        with open('input/domain_deltas_{}.tex'.format(len(inverted_results.keys())), 'w') as f:
            f.write(tex)

    @classmethod
    def false_positive_assessment_table(cls, inverted_results, subset_reindex_keys):
        perfd = defaultdict()
        for subset_str, results in inverted_results.items():
            _rd = {k: v for k, v in results}
            _df = pd.concat([v for _, v in _rd.items()], keys=_rd.keys(), names=['bev', 'var', 't'])
            df_mean = _df.groupby(level='bev').agg(np.nanmean)
            # Take the mean of mean trust values from all other nodes and subtract suspicious node
            perfd[subset_str] = pd.DataFrame(
                [df_mean.drop(x, axis=1).apply(np.nanmean, axis=1) - df_mean[x]
                 for x in df_mean.columns
                 if x != target]) \
                .mean(axis=0)
        perf_df = pd.concat([v for _, v in perfd.items()], keys=perfd.keys(), names=['subset', 'bev']) \
            .unstack('bev').reindex(subset_reindex_keys)
        perf_df['Avg.'] = perf_df.mean(axis=1)
        perf_df = perf_df.append(pd.Series(perf_df.mean(axis=0), name='Avg.'))
        perf_df = perf_df.rename(subset_renamer)
        tex = perf_df.to_latex(float_format=lambda x: "{0:1.2f}".format(x), index=True, column_format="|l|*{4}{c}|r|") \
            .replace('bev', '\diagbox{Domain}{Behaviour}') \
            .split('\n')
        tex.pop(3)  # second dimension header; overridden by the above replacement
        tex.insert(-4, '\hline')  # Hline before averages
        tex = '\n'.join(tex)
        with open('input/domain_deltas_minus.tex', 'w') as f:
            f.write(tex)

    @classmethod
    def true_positive_assessment_table_time_meaned(cls, inverted_results, subset_reindex_keys):
        perfd = defaultdict()
        for subset_str, results in inverted_results.items():
            _rd = {k: v for k, v in results}
            _df = pd.concat([v for _, v in _rd.items()], keys=_rd.keys(), names=['bev', 'var', 't'])
            df_mean = _df.groupby(level='bev').agg(np.nanmean)
            # Take the mean of mean trust values from all other nodes and subtract suspicious node
            perfd[subset_str] = df_mean.drop(target, axis=1).apply(np.nanmean, axis=1) - df_mean[target]
        perf_df = pd.concat([v for _, v in perfd.items()], keys=perfd.keys(), names=['subset', 'bev']) \
            .unstack('bev').reindex(subset_reindex_keys)
        perf_df['Avg.'] = perf_df.mean(axis=1)
        perf_df = perf_df.append(pd.Series(perf_df.mean(axis=0), name='Avg.'))
        perf_df = perf_df.rename(subset_renamer)
        tex = perf_df.to_latex(float_format=lambda x: "{0:1.2f}".format(x), index=True, column_format="|l|*{4}{c}|r|") \
            .replace('bev', '\diagbox{Domain}{Behaviour}') \
            .split('\n')
        tex.pop(3)  # second dimension header; overridden by the above replacement
        tex.insert(-4, '\hline')  # Hline before averages
        tex = '\n'.join(tex)
        with open('input/domain_time_deltas_{}.tex'.format(len(inverted_results.keys())), 'w') as f:
            f.write(tex)

    @classmethod
    def false_positive_assessment_table_time_meaned(cls, inverted_results, subset_reindex_keys):
        perfd = defaultdict()
        for subset_str, results in inverted_results.items():
            _rd = {k: v for k, v in results}
            _df = pd.concat([v for _, v in _rd.items()], keys=_rd.keys(), names=['bev', 'var', 't'])
            df_mean = _df.groupby(level='bev').agg(np.nanmean)
            # Take the mean of mean trust values from all other nodes and subtract suspicious node
            perfd[subset_str] = pd.DataFrame(
                [df_mean.drop(x, axis=1).apply(np.nanmean, axis=1) - df_mean[x]
                 for x in df_mean.columns
                 if x != target]) \
                .mean(axis=0)
        perf_df = pd.concat([v for _, v in perfd.items()], keys=perfd.keys(), names=['subset', 'bev']) \
            .unstack('bev').reindex(subset_reindex_keys)
        perf_df['Avg.'] = perf_df.mean(axis=1)
        perf_df = perf_df.append(pd.Series(perf_df.mean(axis=0), name='Avg.'))
        perf_df = perf_df.rename(subset_renamer)
        tex = perf_df.to_latex(float_format=lambda x: "{0:1.2f}".format(x), index=True, column_format="|l|*{4}{c}|r|") \
            .replace('bev', '\diagbox{Domain}{Behaviour}') \
            .split('\n')
        tex.pop(3)  # second dimension header; overridden by the above replacement
        tex.insert(-4, '\hline')  # Hline before averages
        tex = '\n'.join(tex)
        with open('input/domain_time_deltas_minus_{}.tex'.format(len(inverted_results.keys())), 'w') as f:
            f.write(tex)

    def test(self):
        input_filename = 'input/phys_metric_correlations'
        corrs = calc_correlations_from_weights(self.phys_only_weights)
        with open(input_filename + '.tex', 'w') as f:
            f.write(corrs.loc['Fair'].apply(lambda v: np.round(v, decimals=3)).to_latex(escape=False))
        self.assertTrue(os.path.isfile(input_filename + '.tex'))

    @unittest.skip("GAH")
    def testValidationBoxPlots(self):
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()
        map_levels(trust_observations, var_rename_dict)

        plots = {}
        weights = {}
        for subset_str, key in key_d.items():
            if key is None:
                _trust_observations = trust_observations
            else:
                _trust_observations = trust_observations[key]
            best_d = uncpickle(fig_basedir + '/best_{0}_runs'.format(subset_str))['Fair']
            for target_str, best in best_d.items():
                test_weight = generate_node_trust_perspective(
                    _trust_observations.xs(target_str, level='var'),
                    metric_weights=pd.Series(best[1]))

        fig_filename_prefix = 'boxplots_'
        fig_gen = []

        for (subset_str, target_str), (fig, ax, result) in plots.items():
            fig_filename = "img/box_{0}_run_{1}".format(subset_str, target_str)
            ax.set_title("Example {0} Metric Weighted Assessment for {1}".format(
                subset_renamer(subset_str),
                target_str
            ))
            format_axes(ax)
            savefig(fig, fig_filename, transparent=False)
            self.assertTrue(os.path.isfile(fig_filename + '.' + img_extn))
            if show_outputs:
                try_to_open(fig_filename + '.' + img_extn)
        for fig, ax in fig_gen:
            fig_filename = fig_filename_prefix + ax.get_title().lower().replace(' ', '_') + '.' + img_extn
            savefig(fig_filename)
            self.assertTrue(os.path.isfile(fig_filename))

# if __name__ == '__main__':
#    Aaamas.recompute_features_in_shared(signed=True)
#    Aaamas.recompute_features_in_shared(signed=False)
