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
import pandas as pd
import numpy as np
from collections import OrderedDict
import sklearn.ensemble as ske
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib import rc_context
import matplotlib.ticker as plticker

import sys

import bounos.ChartBuilders as CB

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
show_outputs = True

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6
CB.latexify(columns=2, factor=0.55)

phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']
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

fig_basedir = "/home/bolster/src/thesis/papers/active/16_AAMAS/img"

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

    def testThreatSurfacePlot(self):
        fig_filename = 'img/threat_surface_sum'

        fig, axes = plt.subplots(nrows=1, ncols=3, sharex='none', sharey='none',
                                 subplot_kw={'axisbg': (1, 0.9, 0.9), 'alpha': 0.1})
        # fig.suptitle("Threat Surface for Trust Management Frameworks", size=18, x=0.6, y=0.73)
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
                    xytext=(15, 50), textcoords='offset points',
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
                    xytext=(-25, 30), textcoords='offset points',
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


        fig.savefig(fig_filename, transparent=True)

        self.assertTrue(os.path.isfile(fig_filename+'.png'))

        if show_outputs:
            try_to_open(fig_filename+'.png')

