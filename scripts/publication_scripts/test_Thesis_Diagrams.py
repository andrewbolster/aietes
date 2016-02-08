# coding=utf-8

from __future__ import division

import tempfile
import unittest

import matplotlib.pylab as plt
import pandas as pd
import sklearn.ensemble as ske
from matplotlib import gridspec
from scipy.stats.stats import pearsonr

from aietes.Tools import *
from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.Analyses.Weight import target_weight_feature_extractor, \
    generate_weighted_trust_perspectives, build_outlier_weights, calc_correlations_from_weights, \
    drop_metrics_from_weights_by_key
from bounos.ChartBuilders import format_axes, latexify, unique_cm_dict_from_list
from scripts.publication_scripts import phys_keys, comm_keys, phys_keys_alt, comm_keys_alt, key_order, observer, target, n_metrics, results_path, \
    fig_basedir, subset_renamer

###########
# OPTIONS #
###########
_texcol = 0.5
_texfac = 0.9
use_temp_dir = False
show_outputs = False
recompute = False
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

shared_h5_path = '/dev/shm/shared.h5'
if os.path.exists(shared_h5_path):
    try:
        with pd.get_store(shared_h5_path) as store:
            store.get('joined_target_weights' + "_signed")
    except:
        print("Forcing recompute as the thing I expected to be there, well, isn't")
        recompute = True
else:
    print("Forcing recompute as the thing I expected to be there, well, doesn't exist")
    recompute = True

##################
#  HELPER FUNCS  #
##################

def plot_trust_line_graph(result, title=None, stds=True, spans=None, box=None):

    dropna = lambda x: x[~np.isnan(x)]

    fig_size = latexify(columns=_texcol, factor=_texfac)

    fig = plt.figure(figsize=fig_size)
    #ax = fig.add_subplot(1, 1, 1)
    gs = gridspec.GridSpec(1,2,width_ratios=[4,1])
    ax = plt.subplot(gs[0])
    if spans is not None:
        plottable = pd.stats.moments.ewma(result, span=spans)
    else:
        plottable = result

    def pltstd(tup):
        mean, std = tup
        ax.axhline(mean + std, alpha=0.3, ls=':', color='green')
        ax.axhline(mean - std, alpha=0.3, ls=':', color='red')
    _=plottable.iloc[:,0].plot(ax=ax, alpha=0.8)
    _=plottable.iloc[:,1:].plot(ax=ax, alpha=0.4)

    for i,v in enumerate(result.mean()):
        if i: # Everyone Else
            ax.axhline(v, alpha=0.3, ls='--', color='blue')
        else: # Alfa
            ax.axhline(v, alpha=0.6, ls='-', color='blue')
    if stds:
        map(pltstd, zip(result.mean(), result.std()))
    ax.set_xlabel("Simulated Time (mins)")
    ax.set_ylabel("Weighted Trust Value")
    ax.legend().set_visible(False)
    ax.set_xticks(np.arange(0,61,10))
    ax.set_xticklabels(np.arange(0,61,10))


    if box is not None:
        meanlineprops = dict(linestyle='-', color='blue', alpha=0.8)

        axb = plt.subplot(gs[1])
        if box is 'summary':
            axb.boxplot([dropna(plottable.iloc[:,0].values), plottable.iloc[:,1:].stack().values],
                        labels=['Misbehaver', 'Other Nodes'], widths=0.8, showmeans=True, meanline=True, meanprops=meanlineprops)
        elif box is 'complete':
            plottable.boxplot(rot=90, showmeans=True, meanline=True, meanprops=meanlineprops)
        plt.setp(axb.get_yticklabels(), visible=False)

    fig.tight_layout()

    return fig,[ax, axb],result


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
            plot_trust_line_graph(run, title="{}{}".format(bev_key, run_i))
    return results


def feature_validation_plots(weighted_trust_perspectives, feat_weights, title='Weighted', ewma=True, target='Alfa',
                             observer='Bravo'):
    if ewma:
        _f = lambda f: pd.stats.moments.ewma(f, span=4)
    else:
        _f = lambda f: f

    for key, trust in weighted_trust_perspectives.items():
        fig_size = latexify(columns=_texcol, factor=_texfac)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        _ = _f(trust.unstack('var')[target].xs(observer, level='observer')).boxplot(ax=ax)
        this_title = "{} {}".format(title, '-'.join(key))
        ax.set_title(this_title)
        format_axes(ax)
        fig.tight_layout()
        yield (fig, ax)



assert os.path.isdir(fig_basedir)

def format_features(feats):
    alt_feats = pd.concat(feats, names=['base', 'comp', 'metric']).unstack('metric')
    alt_feats.index.set_levels(
        [[u'MPC', u'STS', u'Fair', u'Shadow', u'SlowCoach'], [u'MPC', u'STS', u'Fair', u'Shadow', u'SlowCoach']],
        inplace=True)
    return alt_feats


class ThesisDiagrams(unittest.TestCase):
    signed = True
    runtime_computed = False

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
        if recompute and self.runtime_computed is not True:
            self.recompute_features_in_shared(signed=self.signed)
            self.runtime_computed = True

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
        for k,v in self.metric_colour_map.items():
            self.metric_colour_map[metric_rename_dict[k]]=v


    @classmethod
    def recompute_features_in_shared(cls, signed=False):
        # All Metrics
        print "Building Joined {} Target Weights".format("Signed" if signed else "Unsigned")
        joined_target_weights = build_outlier_weights(results_path + "/outliers.bkup.h5",
                                                      observer=observer, target=target,
                                                      n_metrics=n_metrics, signed=signed)
        joined_feats = format_features(
            target_weight_feature_extractor(
                joined_target_weights
            )
        )[key_order]
        dumping_suffix = "_signed" if signed else "_unsigned"
        print "Dumping Joined Target Weights"
        joined_target_weights.to_hdf(shared_h5_path, 'joined_target_weights' + dumping_suffix)
        joined_feats.to_hdf(shared_h5_path, 'joined_feats' + dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('comms',comm_keys, joined_target_weights, dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('phys',phys_keys, joined_target_weights, dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('comms_alt',comm_keys_alt, joined_target_weights, dumping_suffix)
        _, _ = cls.metric_subset_weight_and_feature_extractor('phys_alt',phys_keys_alt, joined_target_weights, dumping_suffix)

    @classmethod
    def metric_subset_weight_and_feature_extractor(cls, subset_str, desired_keys, complete_target_weights, dumping_suffix):
        print "Building {} Target Weights".format(subset_str)
        subset_only_weights = drop_metrics_from_weights_by_key(
            complete_target_weights,
            metric_key_inverter(desired_keys)
        )
        subset_only_feats = format_features(
            target_weight_feature_extractor(
                subset_only_weights
            )
        )
        print "Dumping {} Target Weights".format(subset_str)
        subset_only_weights.to_hdf(shared_h5_path, "{}_only_weights{}".format(subset_str,dumping_suffix))
        subset_only_feats.to_hdf(shared_h5_path, "{}_only_feats{}".format(subset_str,dumping_suffix))

        return subset_only_weights, subset_only_feats

    def save_feature_plot(self, feats, fig_filename, hatches=False):
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
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Est. Metric Significance")
        fig = ax.get_figure()
        bars = ax.patches
        if hatches:
            hatches = ''.join(h * 4 for h in ['-', 'x', '\\', '*', 'o', '+', 'O', '.', '_'])
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
        ax.legend(loc='best', ncol=1)
        format_axes(ax)
        ax.xaxis.grid(False) # Disable vertical lines
        fig.tight_layout()
        fig.savefig(fig_filename, transparent=True)
        self.assertTrue(os.path.isfile(fig_filename + '.png'))
        if show_outputs:
            try_to_open(fig_filename + '.png')

    def testThreatSurfacePlot(self):
        fig_filename = 'threat_surface_sum'
        fig_size = latexify(columns=_texcol, factor=_texfac)
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
        fig_filename = 'full_metric_trust_relevance'

        fair_feats = self.joined_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(fair_feats, fig_filename)

    def testCommsMetricTrustRelevance(self):
        fig_filename = 'comms_metric_trust_relevance'

        feats = self.comms_only_feats.loc['Fair'].rename(columns=metric_rename_dict)

        self.save_feature_plot(feats, fig_filename)


    def testPhysMetricTrustRelevance(self):
        fig_filename = 'phys_metric_trust_relevance'

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

    def test0ValidationBestPlots(self):
        """
        :return:
        """
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
            best_d = uncpickle(fig_basedir+'/best_{}_runs'.format(subset_str))['Fair']
            for target_str, best in best_d.items():
                trust_perspective = generate_node_trust_perspective(
                    _trust_observations.xs(target_str, level='var'),
                    metric_weights=pd.Series(best[1]))
                weights[(subset_str,target_str)] = best[1].copy()
                plots[(subset_str,target_str)] = plot_trust_line_graph(trust_perspective \
                                                                     .xs(best[0], level=['observer','run']) \
                                                                     .dropna(axis=1, how='all'),
                                                                       stds=False,
                                                                       spans=6,
                                                                       box='complete')
        inverted_results = defaultdict(list)
        for (subset_str, target_str), (fig,ax,result) in plots.items():
            fig_filename = "best_{}_run_{}".format(subset_str, target_str)
            #ax.set_title("Example {} Metric Weighted Assessment for {}".format(
            #    subset_str.capitalize(),
            #    target_str
            #))
            format_axes(ax)
            fig.savefig(fig_filename, transparent=True)
            self.assertTrue(os.path.isfile(fig_filename + '.png'))
            if show_outputs:
                try_to_open(fig_filename + '.png')
            print subset_str,target_str
            inverted_results[subset_str].append((target_str,result))

        # Dump Best weights from best runs to a table
        w_df = pd.concat([pd.Series(weight, index=key_d[subset.lower()])
                          for (subset,_), weight in weights.items()],
                         keys=weights.keys(),
                         names=['subset','target','metric']).unstack(level='metric')


        # TODO Guaranteed to break (or give really weird results) when new subsets added
        if len(w_df.index.levels[0]) == 3:
            subset_reindex_keys = ['full','comms','phys']
        elif len(w_df.index.levels[0]) == 5:
            subset_reindex_keys = ['full','comms','comms_alt','phys','phys_alt']
        else:
            raise ValueError("Incorrect number of subsets included; {}".format(w_df.index.levels[0]))
        w_df = w_df.reindex(subset_reindex_keys,level='subset')[key_order].rename(columns=metric_rename_dict)
        w_df = w_df.unstack('target').rename(subset_renamer).stack('target')
        tex = w_df.to_latex(float_format=lambda x:"%1.3f"%x, index=True, escape=False, column_format="|l|l|*{{{}}}{{c|}}".format(len(key_order)))\
            .replace('nan','')\
            .split('\n')
        tex[2]='\multicolumn{2}{|c|}{\diagbox{Domain, Behaviour}{Metric}}'+tex[2].lstrip()[1:] # get rid of the pesky field separator
        tex.pop(3) #second dimension header; overridden by the above replacement
        # Add hlines between subsets
        hlines=[]
        for i, line in enumerate(tex):
            begin = line.lstrip()
            if begin and not begin.startswith('&') and not begin.startswith('\\'): # if I'm a subset header line
                hlines.append(i)
        for i,j in enumerate(hlines):
            if i: # first one already covered by midrule
                tex.insert(i+j-1,'\midrule') # add a hline above the i found in prev loop
        tex='\n'.join(tex)
        with open('input/optimised_weights.tex','w') as f:
            f.write(tex)
        print tex

        #mkcpickle("/dev/shm/weights", weights)

        # Generate performance assessments for each weighted metric subset domain against tested behaviour
        perfd = defaultdict()
        for subset_str, results in inverted_results.items():
            _rd = {k:v for k,v in results}
            _df = pd.concat([v for _, v in _rd.items()], keys=_rd.keys(), names=['bev','var','t'])
            df_mean = _df.groupby(level='bev').agg(np.nanmean)
            # Take the mean of mean trust values from all other nodes and subtract suspicious node
            perfd[subset_str] = df_mean.drop(target, axis=1).apply(np.nanmean, axis=1) - df_mean[target]

        perf_df = pd.concat([v for _, v in perfd.items()], keys=perfd.keys(), names=['subset','bev'])\
            .unstack('bev').reindex(subset_reindex_keys)
        perf_df['Avg.']=perf_df.mean(axis=1)
        perf_df = perf_df.append(pd.Series(perf_df.mean(axis=0), name='Avg.'))
        perf_df = perf_df.rename(subset_renamer)

        #mkcpickle("/dev/shm/inverted_results", inverted_results)

        tex=perf_df.to_latex(float_format=lambda x:"%1.2f"%x, index=True, column_format="|l|*{4}{c}|r|")\
            .replace('bev','\diagbox{Domain}{Behaviour}')\
            .split('\n')
        tex.pop(3) #second dimension header; overridden by the above replacement
        tex.insert(-4, '\hline') # Hline before averages
        tex='\n'.join(tex)
        with open('input/domain_deltas.tex','w') as f:
            f.write(tex)
        print tex


        for subset_str, results in inverted_results.items():
            fig_size = latexify(columns=_texcol, factor=_texfac)
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)

            #        _ = _f(trust.unstack('var')[target].xs(observer, level='observer')).boxplot(ax=ax)

            #results.boxplot(ax=ax)
            fig_filename = "box_{}_run_{}".format(subset_str, target_str)
            ax.set_title("Example {} Metric Weighted Assessment for {}".format(
                subset_renamer(subset_str),
                target_str
            ))
            format_axes(ax)
            fig.savefig(fig_filename, transparent=True)
            self.assertTrue(os.path.isfile(fig_filename + '.png'))
            if show_outputs:
                try_to_open(fig_filename + '.png')
            print subset_str,target_str

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
            best_d = uncpickle(fig_basedir+'/best_{}_runs'.format(subset_str))['Fair']
            for target_str, best in best_d.items():
                test_weight = generate_node_trust_perspective(
                    _trust_observations.xs(target_str, level='var'),
                    metric_weights=pd.Series(best[1]))

        fig_filename_prefix = 'boxplots_'
        fig_gen = []

        for (subset_str, target_str), (fig,ax,result) in plots.items():
            fig_filename = "img/box_{}_run_{}".format(subset_str, target_str)
            ax.set_title("Example {} Metric Weighted Assessment for {}".format(
                subset_renamer(subset_str),
                target_str
            ))
            format_axes(ax)
            fig.savefig(fig_filename, transparent=False)
            self.assertTrue(os.path.isfile(fig_filename + '.png'))
            if show_outputs:
                try_to_open(fig_filename + '.png')
            print subset_str,target_str
        for fig, ax in fig_gen:
            fig_filename = fig_filename_prefix + ax.get_title().lower().replace(' ', '_') + '.png'
            fig.savefig(fig_filename)
            self.assertTrue(os.path.isfile(fig_filename))



#if __name__ == '__main__':
#    Aaamas.recompute_features_in_shared(signed=True)
#    Aaamas.recompute_features_in_shared(signed=False)