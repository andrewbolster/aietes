# coding=utf-8

from __future__ import division

import itertools
import tempfile

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import aietes.Tools
from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.Analyses.Weight import summed_outliers_per_weight
from bounos.ChartBuilders import latexify

from scripts.publication_scripts import phys_keys, comm_keys, phys_keys_alt, comm_keys_alt, key_order, observer, target, \
    n_metrics, results_path, \
    fig_basedir, best_of_all


##################
#  HELPER FUNCS  #
##################
def plot_result(result, title=None, stds=True):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    result.plot(ax=ax)
    pltlin = lambda v: ax.axhline(v, alpha=0.3, color='blue')

    def pltstd(tup):
        mean, std = tup
        ax.axhline(mean + std, alpha=0.3, color='green')
        ax.axhline(mean - std, alpha=0.3, color='red')

    map(pltlin, result.mean())
    if stds:
        map(pltstd, zip(result.mean(), result.std()))
    ax.set_title(title)
    print(result.describe())
    plt.show()


###########
# OPTIONS #
###########
use_temp_dir = False
show_outputs = False
recompute = False
shared_h5_path = '/dev/shm/shared.h5'

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6
latexify(columns=2, factor=0.55)

# These were arrived at through visual interpretation of the complete metric relevancy graphs

assert aietes.Tools.os.path.isdir(fig_basedir)


class AaamasResultSelection(object):
    signed = True

    def __init__(self):
        if use_temp_dir:
            self.dirpath = tempfile.mkdtemp()
        else:
            self.dirpath = fig_basedir

        aietes.Tools.os.chdir(self.dirpath)

        if not aietes.Tools.os.path.exists("img"):
            aietes.Tools.os.makedirs("img")
        if not aietes.Tools.os.path.exists("input"):
            aietes.Tools.os.makedirs("input")

        # # Plot Style Config

        _boxplot_kwargs = {
            'showmeans': True,
            'showbox': False,
            'widths': 0.2,
            'linewidth': 2
        }

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

        self.joined_feat_weights = aietes.Tools.categorise_dataframe(aietes.Tools.non_zero_rows(self.joined_feats).T)
        self.comms_feat_weights = aietes.Tools.categorise_dataframe(aietes.Tools.non_zero_rows(self.comms_only_feats).T)
        self.phys_feat_weights = aietes.Tools.categorise_dataframe(aietes.Tools.non_zero_rows(self.phys_only_feats).T)
        self.comms_alt_feat_weights = aietes.Tools.categorise_dataframe(
            aietes.Tools.non_zero_rows(self.comms_alt_only_feats).T)
        self.phys_alt_feat_weights = aietes.Tools.categorise_dataframe(
            aietes.Tools.non_zero_rows(self.phys_alt_only_feats).T)

        print("Got Everything I Need!")
        self.testGetBestFullRuns()

    def testGetBestFullRuns(self):
        """
        Purpose of this is to get the best results for full-metric scope
        (as defined as max(T_~Alfa.mean() - T_Alfa.mean())
        A "Run" is from
            an individual node on an individual run
        This returns a run for
            each non-control behaviour

        i.e. something that will be sensibly processed by
        _inner = lambda x: map(np.nanmean,np.split(x, [1], axis=1))
        assess = lambda x: -np.subtract(*_inner(x))
        assess_run = lambda x: assess(x.xs(target_str, level='var').xs(0,level='run').values)

        :return: best run
        """
        feat_d = {
            'full': (self.joined_feat_weights, key_order),
            # 'comms':(self.comms_feat_weights, comm_keys),
            # 'phys': (self.phys_feat_weights, phys_keys),
            # 'comms_alt': (self.comms_alt_feat_weights, comm_keys_alt),
            # 'phys_alt': (self.phys_alt_feat_weights, phys_keys_alt),
        }
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()
        for feat_str, (feats, keys) in feat_d.items():
            print(feat_str)
            if keys is not None:
                best = best_of_all(feats, trust_observations[keys])
            else:
                best = best_of_all(feats, trust_observations)
            aietes.Tools.mkcpickle('best_{0}_runs'.format(feat_str), dict(best))


if __name__ == "__main__":
    AaamasResultSelection()
