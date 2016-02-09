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

from scripts.publication_scripts import phys_keys, comm_keys, phys_keys_alt, comm_keys_alt, key_order, observer, target, n_metrics, results_path, \
    fig_basedir

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


def build_outlier_weights(h5_path, signed=False):
    """Outliers should have keys of runs
    :param h5_path:
    """
    with pd.get_store(h5_path) as store:
        keys = store.keys()

    target_weights_dict = {}
    for runkey in filter(lambda s: s.startswith('/CombinedTrust'), keys):
        with pd.get_store(h5_path) as store:
            print runkey
            target_weights_dict[runkey] = summed_outliers_per_weight(store.get(runkey),
                                                                     observer, n_metrics,
                                                                     target=target,
                                                                     signed=False)

    joined_target_weights = pd.concat(
            target_weights_dict, names=['run'] + target_weights_dict[runkey].index.names
    ).reset_index('run', drop=True).sort()

    return joined_target_weights


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
        self.comms_alt_feat_weights = aietes.Tools.categorise_dataframe(aietes.Tools.non_zero_rows(self.comms_alt_only_feats).T)
        self.phys_alt_feat_weights = aietes.Tools.categorise_dataframe(aietes.Tools.non_zero_rows(self.phys_alt_only_feats).T)

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
            #'comms':(self.comms_feat_weights, comm_keys),
            #'phys': (self.phys_feat_weights, phys_keys),
            #'comms_alt': (self.comms_alt_feat_weights, comm_keys_alt),
            #'phys_alt': (self.phys_alt_feat_weights, phys_keys_alt),
        }
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()
        for feat_str, (feats, keys) in feat_d.items():
            print(feat_str)
            if keys is not None:
                best = self.best_of_all(feats, trust_observations[keys])
            else:
                best = self.best_of_all(feats, trust_observations)
            aietes.Tools.mkcpickle('best_{}_runs'.format(feat_str), dict(best))

    def best_of_all(self, feats, trust_observations):
        best = aietes.Tools.defaultdict(dict)
        for (base_str, target_str), feat in feats.to_dict().items():
            if base_str != "Fair":
                continue
            print(base_str)

            print("---" + target_str)
            best[base_str][target_str] = \
                self.best_run_and_weight(
                        feat,
                        trust_observations)
        return best

    def best_run_and_weight(self, f, trust_observations, par=True, tolerance=0.01):
        f = pd.Series(f, index=trust_observations.keys())
        f_val = f.values

        def _assess(x):
            return -np.subtract(*map(np.nanmean, np.split(x.values, [1], axis=1)))

        @aietes.Tools.timeit()
        def generate_weighted_trust_perspectives(_trust_observations, feat_weights, par=True):
            weighted_trust_perspectives = []
            for w in feat_weights:
                weighted_trust_perspectives.append(
                        generate_node_trust_perspective(
                                _trust_observations,
                                metric_weights=pd.Series(w),
                                par=par
                        ))
            return weighted_trust_perspectives

        def best_group_in_perspective(perspective):
            group = perspective.groupby(level=['observer', 'run']) \
                .apply(_assess)
            best_group = group.argmax()
            return best_group, group[best_group]

        combinations = np.asarray([f_val * i for i in itertools.product([-1, 1], repeat=len(f))])
        for i in f.values[np.abs(f_val) < tolerance]:
            combinations[:, np.where(f_val == i)] = i
        combinations = aietes.Tools.npuniq(combinations)

        print("Have {} Combinations".format(len(combinations)))
        perspectives = generate_weighted_trust_perspectives(trust_observations,
                                                            combinations, par=par)
        print("Got Perspectives")
        group_keys, assessments = zip(*map(best_group_in_perspective, perspectives))
        best_weight = combinations[np.argmax(assessments)]
        best_run = group_keys[np.argmax(assessments)]
        best_score = np.max(assessments)
        print("Winner is {} with {}@{}".format(best_run, best_weight, best_score))
        if np.all(best_weight == f):
            print("Actually got it right first time for a change!")
        return best_run, best_weight


if __name__ == "__main__":
    AaamasResultSelection()
