# coding=utf-8

from __future__ import division
import itertools
import tempfile
import unittest
from collections import defaultdict
import matplotlib.pylab as plt
import pandas as pd
import sklearn.ensemble as ske
from scipy.stats.stats import pearsonr

from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.ChartBuilders import format_axes, latexify
from aietes.Tools import *
from bounos.Analyses.Weight import summed_outliers_per_weight, target_weight_feature_extractor, \
    generate_weighted_trust_perspectives


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


def assess_result(result, target='Alfa'):
    m = result.drop(target, 1).mean().mean() - result[target].mean()
    std = result.std().mean()
    return m, std


def assess_results(perspectives_d, base_key='Fair'):
    keys = map(lambda k: k[1], filter(lambda k: k[0] == base_key, perspectives_d.keys()))
    results = defaultdict(dict)
    for bev_key in keys:
        for run_i, run in perspectives_d[(base_key, bev_key)].xs(bev_key, level='var').groupby(level='run'):
            results[bev_key][run_i] = assess_result(run)
            plot_result(run, title="{}{}".format(bev_key, run_i))
    return results


def feature_validation_plots(weighted_trust_perspectives, feat_weights, title='Weighted', ewma=True, target='Alfa',
                             observer='Bravo'):
    if ewma:
        _f = lambda f: pd.stats.moments.ewma(f, span=4)
    else:
        _f = lambda f: f

    for key, trust in weighted_trust_perspectives.items():
        fig_size = latexify(columns=0.5, factor=1)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        _ = _f(trust.unstack('var')[target].xs(observer, level='observer')).boxplot(ax=ax)
        this_title = "{} {}".format(title, '-'.join(key))
        ax.set_title(this_title)
        format_axes(ax)
        fig.tight_layout(pad=0.3)
        yield (fig, ax)


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

phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']

key_order = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR', 'INDD', 'INHD', 'Speed']

observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
shared_h5_path = '/dev/shm/shared.h5'

fig_basedir = "/home/bolster/src/thesis/papers/active/16_AAMAS"

assert os.path.isdir(fig_basedir)


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
        :param x_width:
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

    corrs = pd.DataFrame.from_dict(_corrs).T.rename(columns=metric_rename_dict)
    map_levels(corrs, var_rename_dict, 0)
    map_levels(corrs, var_rename_dict, 1)
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
    signed = True

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
            self.recompute_features_in_shared(signed=self.signed)

        dumping_suffix = "_signed" if self.signed else "_unsigned"
        with pd.get_store(shared_h5_path) as store:
            self.joined_target_weights = store.get('joined_target_weights' + dumping_suffix)
            self.joined_feats = store.get('joined_feats' + dumping_suffix)
            self.comms_only_feats = store.get('comms_only_feats' + dumping_suffix)
            self.phys_only_feats = store.get('phys_only_feats' + dumping_suffix)
            self.comms_only_weights = store.get('comms_only_weights' + dumping_suffix)
            self.phys_only_weights = store.get('phys_only_weights' + dumping_suffix)

        self.joined_feat_weights = categorise_dataframe(non_zero_rows(self.joined_feats).T)
        self.comms_feat_weights = categorise_dataframe(non_zero_rows(self.comms_only_feats).T)
        self.phys_feat_weights = categorise_dataframe(non_zero_rows(self.phys_only_feats).T)

    @classmethod
    def recompute_features_in_shared(cls, signed=False):
        # All Metrics
        print "Building Joined {} Target Weights".format("Signed" if signed else "Unsigned")
        joined_target_weights = build_outlier_weights(results_path + "/outliers.bkup.h5", signed=signed)
        joined_feats = format_features(
            target_weight_feature_extractor(
                joined_target_weights
            )
        )
        dumping_suffix = "_signed" if signed else "_unsigned"
        print "Dumping Joined Target Weights"
        joined_target_weights.to_hdf(shared_h5_path, 'joined_target_weights' + dumping_suffix)
        joined_feats.to_hdf(shared_h5_path, 'joined_feats' + dumping_suffix)
        print "Building Comms Target Weights"
        comms_only_weights = drop_metrics_from_weights_by_key(
            joined_target_weights,
            phys_keys
        )
        comms_only_feats = format_features(
            target_weight_feature_extractor(
                comms_only_weights
            )
        )
        print "Dumping Comms Target Weights"
        comms_only_weights.to_hdf(shared_h5_path, 'comms_only_weights' + dumping_suffix)
        comms_only_feats.to_hdf(shared_h5_path, 'comms_only_feats' + dumping_suffix)
        print "Building Phys Target Weights"
        phys_only_weights = drop_metrics_from_weights_by_key(
            joined_target_weights,
            comm_keys
        )
        phys_only_feats = format_features(
            target_weight_feature_extractor(
                phys_only_weights
            )
        )
        print "Dumping Phys Target Weights"
        phys_only_weights.to_hdf(shared_h5_path, 'phys_only_weights' + dumping_suffix)
        phys_only_feats.to_hdf(shared_h5_path, 'phys_only_feats' + dumping_suffix)

    def testThreatSurfacePlot(self):
        fig_filename = 'img/threat_surface_sum'
        fig_size = latexify(columns=0.5, factor=0.9)
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

        axes = map(format_axes, axes)
        fig.delaxes(axes[3])
        fig.savefig(fig_filename, transparent=False)

        self.assertTrue(os.path.isfile(fig_filename + '.png'))

        if show_outputs:
            try_to_open(fig_filename + '.png')

    def testFullMetricTrustRelevance(self):
        fig_filename = 'img/full_metric_trust_relevance'

        fair_feats = self.joined_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(fair_feats, fig_filename)

    def testCommsMetricTrustRelevance(self):
        fig_filename = 'img/comms_metric_trust_relevance'

        feats = self.comms_only_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(feats, fig_filename)

    def save_feature_plot(self, feats, fig_filename):
        fig_size = latexify(columns=0.5, factor=1)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
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
        format_axes(ax)
        fig.tight_layout(pad=0.3)
        fig.savefig(fig_filename, transparent=True)
        self.assertTrue(os.path.isfile(fig_filename + '.png'))
        if show_outputs:
            try_to_open(fig_filename + '.png')

    def testPhysMetricTrustRelevance(self):
        fig_filename = 'img/phys_metric_trust_relevance'

        feats = self.phys_only_feats.loc['Fair'].rename(columns=metric_rename_dict)
        self.save_feature_plot(feats, fig_filename)

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
            print scores, sp.stats.describe(scores)

    def test0ValidationBoxPlots(self):
        """

        :return:
        """
        subset_str = "Full"
        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.xs('Bravo', level='observer', drop_level=False).dropna()
        feat_weights_d = {
            "full": self.joined_feat_weights,
            "comms only": self.comms_feat_weights,
            "phys only": self.phys_feat_weights
        }

        for subset_str, feat_weights in feat_weights_d.items():
            self.boxPlotValidations(trust_observations, feat_weights, subset_str)

    def boxPlotValidations(self, trust_observations, feat_weights, subset_str):
        fig_filename_prefix = 'img/boxplots_'
        perspectives = generate_weighted_trust_perspectives(trust_observations, feat_weights)
        fig_gen = feature_validation_plots(perspectives,
                                           feat_weights,
                                           title="{} {} Weighted".format(
                                               subset_str.capitalize(),
                                               "Signed" if self.signed else "Unsigned"
                                           ))
        for fig, ax in fig_gen:
            fig_filename = fig_filename_prefix + ax.get_title().lower().replace(' ', '_') + '.png'
            fig.savefig(fig_filename)
            self.assertTrue(os.path.isfile(fig_filename))


class AaamasResultSelection(unittest.TestCase):
    signed = True

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

        dumping_suffix = "_signed" if self.signed else "_unsigned"
        with pd.get_store(shared_h5_path) as store:
            self.joined_target_weights = store.get('joined_target_weights' + dumping_suffix)
            self.joined_feats = store.get('joined_feats' + dumping_suffix)
            self.comms_only_feats = store.get('comms_only_feats' + dumping_suffix)
            self.phys_only_feats = store.get('phys_only_feats' + dumping_suffix)
            self.comms_only_weights = store.get('comms_only_weights' + dumping_suffix)
            self.phys_only_weights = store.get('phys_only_weights' + dumping_suffix)

        self.joined_feat_weights = categorise_dataframe(non_zero_rows(self.joined_feats).T)
        self.comms_feat_weights = categorise_dataframe(non_zero_rows(self.comms_only_feats).T)
        self.phys_feat_weights = categorise_dataframe(non_zero_rows(self.phys_only_feats).T)

    def testGetBestFullRuns(self):
        """
        Purpose of this is to get the best results for full-metric scope
        (as defined as max(T_~Alfa.mean() - T_Alfa.mean())

        :return:
        """
        pass



    def testGetBestCommsRuns(self):
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


        def drop_metrics_from_weights_by_key(target_weights, drop_keys):
            reset_by_keys = target_weights.reset_index(level=drop_keys)
            zero_indexes = (reset_by_keys[drop_keys] == 0.0).all(axis=1)
            dropped_target_weights = reset_by_keys[zero_indexes].drop(drop_keys, 1)
            return dropped_target_weights

        with pd.get_store(results_path + '.h5') as store:
            trust_observations = store.trust.dropna()



        comms_weights = drop_metrics_from_weights_by_key(self.joined_target_weights, ['INDD', 'INHD', 'Speed'] )
        feats = target_weight_feature_extractor(comms_weights, raw=True)

        available_bev_keys = feats.keys()
        base_str = 'CombinedTrust'
        target_str = 'CombinedSelfishTargetSelection'
        for target, reg in feats[base_str]:
            if target == target_str:
                f = reg.feature_importances_


        _inner = lambda x: map(np.nanmean,np.split(x, [1], axis=1))
        assess = lambda x: -np.subtract(*_inner(x))

        def generate_weighted_trust_perspectives(trust_observations, feat_weights, par=True):
            weighted_trust_perspectives = []
            for w in feat_weights:
                weighted_trust_perspectives.append(generate_node_trust_perspective(
                    trust_observations,
                    metric_weights=pd.Series(w),
                    par=par
                ))

            return weighted_trust_perspectives


        def best_group_in_perspective(perspective):
            group = perspective.groupby(level=['observer','run']).apply(assess)
            best_group = group.argmax()
            return best_group, group[best_group]

        combinations = [f * i for i in itertools.product([-1,1],repeat=6)]

        perspectives = generate_weighted_trust_perspectives(trust_observations[comm_keys],
                                                        combinations, par=True)
        group_keys, assessments = zip(*map(best_group_in_perspective,perspectives))


        best_weight = combinations[np.argmax(assessments)]
        best_run = group_keys[np.argmax(assessments)]
        if np.all(best_weight == f):
            print("Actually got it right first time for a change!")
        self.assertEqual(('Foxtrot', 0),best_run)
        return best_weight, best_run






#if __name__ == '__main__':
#    Aaamas.recompute_features_in_shared(signed=True)
#    Aaamas.recompute_features_in_shared(signed=False)
