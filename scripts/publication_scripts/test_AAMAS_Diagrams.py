# coding=utf-8

from __future__ import division
import subprocess
import unittest
import operator
import os
import warnings
import traceback
from os.path import expanduser
import tempfile
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.stats.stats import pearsonr
import sklearn.ensemble as ske
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib import rc_context
import matplotlib.ticker as plticker
import sys
import bounos.ChartBuilders as CB
from bounos.Analyses.Weight import summed_outliers_per_weight


##################
#  HELPER FUNCS  #
##################

def categorise_dataframe(df):
    # Categories work better as indexes
    for obj_key in df.keys()[df.dtypes == object]:
        try:
            df[obj_key] = df[obj_key].astype('category')
        except TypeError:
            print("Couldn't categorise {}".format(obj_key))
            pass
    return df


def feature_extractor(df, target):
    data = df.drop(target, axis=1)
    reg = ske.RandomForestRegressor(n_jobs=4, n_estimators=512)
    reg.fit(data, df[target])
    return pd.Series(dict(zip(data.keys(), reg.feature_importances_)))


def target_weight_feature_extractor(target_weights):
    known_good_features_d = {}
    for basekey in target_weights.keys():  # Parallelisable
        print basekey
        # Single DataFrame of all features against one behaviour
        var_weights = target_weights.apply(lambda s: s / target_weights[basekey], axis=0).dropna()
        known_good_features_d[basekey] = \
            pd.concat([feature_extractor(s.reset_index(), var) for var, s in var_weights.iteritems()],
                      keys=var_weights.keys(), names=['var', 'metric'])

    return known_good_features_d


def dataframe_weight_filter(df, keys):
    indexes = [(df.index.get_level_values(k) == 0.0) for k in keys]
    return df.loc[reduce(operator.and_, indexes)]


###########
# OPTIONS #
###########
use_temp_dir = False
show_outputs = False
recompute = True
shared_h5_path = '/dev/shm/shared.h5'

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6
CB.latexify(columns=2, factor=0.55)

phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']

key_order = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR', 'INDD', 'INHD', 'Speed']

observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9

var_rename_dict = {'CombinedBadMouthingPowerControl': 'MPC',
                   'CombinedSelfishTargetSelection': 'STS',
                   'CombinedTrust': 'Fair',
                   'Shadow': 'Shadow',
                   'SlowCoach': 'SlowCoach'}

metric_rename_dict = {
    'ADelay': "$Delay$",
    'ARXP': "$P_{RX}$",
    'ATXP': "$P_{TX}$",
    'RXThroughput': "$T^P_{RX}$",
    'TXThroughput': "$T^P_{TX}$",
    'PLR': '$PLR$',
    'INDD': '$INDD$',
    'INHD': '$INHD$',
    'Speed': '$Speed$'
}

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
shared_h5_path = '/dev/shm/shared.h5'

fig_basedir = "/home/bolster/src/thesis/papers/active/16_AAMAS"

assert os.path.isdir(fig_basedir)


def try_to_open(filename):
    try:
        if sys.platform == 'linux2':
            subprocess.call(["xdg-open", filename])
        else:
            os.startfile(filename)
    except Exception:
        warnings.warn(traceback.format_exc())


def non_zero_rows(df):
    return df[~(df == 0).all(axis=1)]


def map_level(df, dct, level=0):
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i == level else names
                      for i, names in enumerate(index.levels)], inplace=True)


def add_height_annotation(ax, start, end, txt_str, x_width=.5, txt_kwargs=None, arrow_kwargs=None):
    """
    Adds horizontal arrow annotation with text in the middle

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to draw to

    start : float
        start of line

    end : float
        end of line

    txt_str : string
        The text to add

    y_height : float
        The height of the line

    txt_kwargs : dict or None
        Extra kwargs to pass to the text

    arrow_kwargs : dict or None
        Extra kwargs to pass to the annotate

    Returns
    -------
    tuple
        (annotation, text)
    """

    if txt_kwargs is None:
        txt_kwargs = {}
    if arrow_kwargs is None:
        # default to your arrowprops
        arrow_kwargs = {'arrowprops': dict(arrowstyle="<->",
                                           connectionstyle="bar",
                                           ec="k",
                                           shrinkA=5, shrinkB=5,
                                           )}

    trans = ax.get_xaxis_transform()

    ann = ax.annotate('', xy=(x_width, start),
                      xytext=(x_width, end),
                      transform=trans,
                      **arrow_kwargs)
    txt = ax.text(x_width + .05,
                  (start + end) / 2,
                  txt_str,
                  **txt_kwargs)


def build_target_weights(h5_path):
    """Outliers should have keys of runs"""
    with pd.get_store(h5_path) as store:
        target_weights_dict = {}
        for runkey in store.keys():
            print runkey
            target_weights_dict[runkey] = summed_outliers_per_weight(store.get(runkey), observer, n_metrics,
                                                                     target=target)

    joined_target_weights = pd.concat(target_weights_dict, names=['run'] + target_weights_dict[runkey].index.names)
    sorted_joined_target_weights = joined_target_weights.reset_index('run', drop=True).sort()

    return sorted_joined_target_weights


def calc_correlations(base, comp, index=0):
    dp_r = (comp / base).reset_index()
    return dp_r.corr()[index][:-1]


def calc_correlations_from_weights(weights):
    _corrs = {}
    for base, comp in itertools.permutations(weights.keys(), 2):
        _corrs[(base, comp)] = \
            calc_correlations(weights[base],
                              weights[comp])

    corrs = pd.DataFrame.from_dict(_corrs).T.rename(columns=metric_rename_dict)
    map_level(corrs, var_rename_dict, 0)
    map_level(corrs, var_rename_dict, 1)
    corrs.index.set_names(['Control', 'Misbehaviour'], inplace=True)
    return corrs


def drop_metrics_from_weights_by_key(target_weights, drop_keys):
    reset_by_keys = target_weights.reset_index(level=drop_keys)
    zero_indexes = (reset_by_keys[drop_keys] == 0.0).all(axis=1)
    dropped_target_weights = reset_by_keys[zero_indexes].drop(drop_keys, 1)
    return dropped_target_weights


def format_features(feats):
    alt_feats = pd.concat(feats, names=['base', 'comp', 'metric']).unstack('metric')
    alt_feats.index.set_levels(
        [[u'MPC', u'STS', u'Fair', u'Shadow', u'SlowCoach'], [u'MPC', u'STS', u'Fair', u'Shadow', u'SlowCoach']],
        inplace=True)
    return alt_feats


class Aaamas(unittest.TestCase):
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
        if recompute:
            # All Metrics
            self.joined_target_weights = build_target_weights(results_path + "/outliers.bkup.h5")
            self.joined_feats = format_features(
                target_weight_feature_extractor(
                    self.joined_target_weights
                )
            )

            self.joined_target_weights.to_hdf(shared_h5_path, 'joined_target_weights')
            self.joined_feats.to_hdf(shared_h5_path, 'joined_feats')

            self.comms_only_weights = drop_metrics_from_weights_by_key(
                self.joined_target_weights,
                phys_keys
            )
            self.comms_only_feats = format_features(
                target_weight_feature_extractor(
                    self.comms_only_weights
                )
            )
            self.comms_only_weights.to_hdf(shared_h5_path, 'comms_only_weights')

            self.comms_only_feats.to_hdf(shared_h5_path, 'comms_only_feats')


            self.phys_only_weights = drop_metrics_from_weights_by_key(
                self.joined_target_weights,
                comm_keys
            )
            self.phys_only_feats = format_features(
                target_weight_feature_extractor(
                    self.phys_only_weights
                )
            )
            self.phys_only_weights.to_hdf(shared_h5_path, 'phys_only_weights')
            self.phys_only_feats.to_hdf(shared_h5_path, 'phys_only_feats')

        else:
            with pd.get_store(shared_h5_path) as store:
                self.joined_target_weights = store.get('joined_target_weights')
                self.joined_feats = store.get('joined_feats')
                self.comms_only_feats = store.get('comms_only_feats')
                self.phys_only_feats = store.get('phys_only_feats')
                self.comms_only_weights = store.get('comms_only_weights')
                self.phys_only_weights = store.get('phys_only_weights')

    def testThreatSurfacePlot(self):
        fig_filename = 'img/threat_surface_sum'
        fig_size = CB.latexify(columns=0.5, factor=0.9)
        fig_size = (fig_size[0], fig_size[1] / 2)
        print fig_size

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
        ax.set_title("Single Vector")
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

        axes = map(CB.format_axes, axes)
        fig.delaxes(axes[3])
        fig.savefig(fig_filename, transparent=False)

        self.assertTrue(os.path.isfile(fig_filename + '.png'))

        if show_outputs:
            try_to_open(fig_filename + '.png')

    def testFullMetricTrustRelevance(self):
        fig_filename = 'img/full_metric_trust_relevance'
        fig_size = CB.latexify(columns=0.5, factor=1)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)

        fair_feats = self.joined_feats.loc['Fair'].rename(columns=metric_rename_dict)

        ax = fair_feats[~(fair_feats == 0).all(axis=1)].plot(
            ax=ax, kind='bar', rot=0, width=0.9, figsize=fig_size,
            legend=False
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Est. Best Metric Weighting")
        fig = ax.get_figure()

        bars = ax.patches
        hatches = ''.join(h * 4 for h in ['-', 'x', '\\', '*', 'o', '+', 'O', '.', '_'])

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.legend(loc='best', ncol=1)
        CB.format_axes(ax)

        fig.tight_layout(pad=0.3)
        fig.savefig(fig_filename, transparent=True)
        self.assertTrue(os.path.isfile(fig_filename + '.png'))
        if show_outputs:
            try_to_open(fig_filename + '.png')

    def testCommsMetricTrustRelevance(self):
        fig_filename = 'img/comms_metric_trust_relevance'
        fig_size = CB.latexify(columns=0.5, factor=1)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)

        feats = self.comms_only_feats.loc['Fair'].rename(columns=metric_rename_dict)

        ax = feats[~(feats == 0).all(axis=1)].plot(
            ax=ax, kind='bar', rot=0, width=0.9, figsize=fig_size,
            legend=False
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Est. Best Metric Weighting")
        fig = ax.get_figure()

        bars = ax.patches
        hatches = ''.join(h * 4 for h in ['-', 'x', '\\', '*', 'o', '+', 'O', '.', '_'])

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.legend(loc='best', ncol=1)
        CB.format_axes(ax)

        fig.tight_layout(pad=0.3)
        fig.savefig(fig_filename, transparent=True)
        self.assertTrue(os.path.isfile(fig_filename + '.png'))
        if show_outputs:
            try_to_open(fig_filename + '.png')

    def testPhysMetricTrustRelevance(self):
        fig_filename = 'img/phys_metric_trust_relevance'
        fig_size = CB.latexify(columns=0.5, factor=1)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)

        feats = self.phys_only_feats.loc['Fair'].rename(columns=metric_rename_dict)

        ax = feats[~(feats == 0).all(axis=1)].plot(
            ax=ax, kind='bar', rot=0, width=0.9, figsize=fig_size,
            legend=False
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Est. Best Metric Weighting")
        fig = ax.get_figure()

        bars = ax.patches
        hatches = ''.join(h * 4 for h in ['-', 'x', '\\', '*', 'o', '+', 'O', '.', '_'])

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.legend(loc='best', ncol=1)
        CB.format_axes(ax)

        fig.tight_layout(pad=0.3)
        fig.savefig(fig_filename, transparent=True)
        self.assertTrue(os.path.isfile(fig_filename + '.png'))
        if show_outputs:
            try_to_open(fig_filename + '.png')

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
