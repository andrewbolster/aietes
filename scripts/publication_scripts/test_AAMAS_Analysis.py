# coding=utf-8

from __future__ import division
import itertools
import tempfile
import unittest
import matplotlib.pylab as plt
import pandas as pd

from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.ChartBuilders import format_axes, latexify
import aietes.Tools
from bounos.Analyses.Weight import summed_outliers_per_weight, target_weight_feature_extractor


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

_ = aietes.Tools.np.seterr(invalid='ignore')  # Pandas PITA Nan printing

golden_mean = (aietes.Tools.np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6
latexify(columns=2, factor=0.55)

phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']

key_order = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR', 'INDD', 'INHD', 'Speed']

observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
fig_basedir = "/home/bolster/src/thesis/papers/active/16_AAMAS"

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


def build_mean_delta_t_weights(h5_path):
    with pd.get_store(h5_path) as store:
        mdts = store.get('meandeltaCombinedTrust_2')


def calc_correlations_from_weights(weights):
    def calc_correlations(base, comp, index=0):
        dp_r = (comp / base).reset_index()
        return dp_r.corr()[index][:-1]

    _corrs = {}
    for base, comp in itertools.permutations(weights.keys(), 2):
        _corrs[(base, comp)] = \
            calc_correlations(weights[base],
                              weights[comp])

    corrs = pd.DataFrame.from_dict(_corrs).T.rename(columns=aietes.Tools.metric_rename_dict)
    aietes.Tools.map_levels(corrs, aietes.Tools.var_rename_dict, 0)
    aietes.Tools.map_levels(corrs, aietes.Tools.var_rename_dict, 1)
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


class AaamasResultSelection(unittest.TestCase):
    signed = True

    @classmethod
    def setUpClass(cls):
        if use_temp_dir:
            cls.dirpath = tempfile.mkdtemp()
        else:
            cls.dirpath = fig_basedir

        aietes.Tools.os.chdir(cls.dirpath)

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

        dumping_suffix = "_signed" if cls.signed else "_unsigned"
        with pd.get_store(shared_h5_path) as store:
            cls.joined_target_weights = store.get('joined_target_weights' + dumping_suffix)
            cls.joined_feats = store.get('joined_feats' + dumping_suffix)

        cls.joined_feat_weights = aietes.Tools.categorise_dataframe(aietes.Tools.non_zero_rows(cls.joined_feats).T)

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

        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()

        feats = target_weight_feature_extractor(self.joined_target_weights, raw=True)

        best = self.best_of_all(feats, trust_observations)

        aietes.Tools.mkcpickle('best_runs', dict(best))
        self.assertTrue()

    def testGetBestCommsRuns(self):
        """
        Purpose of this is to get the best results for comms-metric scope
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

        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()

        print("Got Trust")

        comms_weights = drop_metrics_from_weights_by_key(self.joined_target_weights, phys_keys)
        feats = target_weight_feature_extractor(comms_weights, raw=True)

        best = self.best_of_all(feats, trust_observations)

        aietes.Tools.mkcpickle('best_comms_runs', dict(best))
        self.assertEqual(('Foxtrot', 0),
                         best['CombinedTrust']['CombinedSelfishTargetSelection'][0])

    def best_of_all(self, feats, trust_observations):
        best = aietes.Tools.defaultdict(dict)
        for base_str, reg_list in feats.items():
            print(base_str)
            if base_str != "CombinedTrust":
                continue
            for target_str, reg in reg_list:
                if target_str == base_str:
                    continue
                else:
                    print("---" + target_str)
                    best[base_str][target_str] = \
                        self.best_run_and_weight(
                            reg.feature_importances_,
                            trust_observations)
        return best

    def testGetBestPhysRuns(self):
        """
        Purpose of this is to get the best results for phys-metric scope
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
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()

        phys_weights = drop_metrics_from_weights_by_key(self.joined_target_weights, comm_keys)
        feats = target_weight_feature_extractor(phys_weights, raw=True)

        best = self.best_of_all(feats, trust_observations)


        aietes.Tools.mkcpickle('best_phys_runs', dict(best))
        self.assertTrue(True)

    def best_run_and_weight(self, f, trust_observations):

        def generate_weighted_trust_perspectives(_trust_observations, feat_weights, par=True):
            weighted_trust_perspectives = []
            for w in feat_weights:
                weighted_trust_perspectives.append(generate_node_trust_perspective(
                    _trust_observations,
                    metric_weights=pd.Series(w),
                    par=par
                ))
            return weighted_trust_perspectives

        def best_group_in_perspective(perspective):
            group = perspective.groupby(level=['observer', 'run'])\
                .apply(lambda x: -aietes.Tools.np.subtract(*map(aietes.Tools.np.nanmean, aietes.Tools.np.split(x, [1], axis=1))))
            best_group = group.argmax()
            return best_group, group[best_group]

        combinations = [f * i for i in itertools.product([-1, 1], repeat=len(f))]
        perspectives = generate_weighted_trust_perspectives(trust_observations[comm_keys],
                                                            combinations, par=False)
        group_keys, assessments = zip(*map(best_group_in_perspective, perspectives))
        best_weight = combinations[aietes.Tools.np.argmax(assessments)]
        best_run = group_keys[aietes.Tools.np.argmax(assessments)]
        if aietes.Tools.np.all(best_weight == f):
            print("Actually got it right first time for a change!")
        return best_run, best_weight