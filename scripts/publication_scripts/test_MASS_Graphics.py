import unittest
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from bounos.ChartBuilders import plot_axes_views_from_positions_frame
from matplotlib import pylab as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from aietes.Tools import map_levels
from bounos.Analyses.Dixon import  *



class smart_dict(dict):
    # Replaces missing lookups with the key
    def __missing__(self, key):
        return key

def plot_behaviour_metric_graph(data, fig_path, observer = 'Bravo', target='Alfa', run = 0,
                                force_ymin=None, title=None, y_label_map=None,
                                save_fig=False, show_title=False):

    _bevs = set(data.index.get_level_values('var'))
    _metrics = list(data.columns)
    fig = plt.figure(figsize=(8, 5), dpi=80)
    base_ax = fig.add_axes([0, 0, 1, 1], )
    base_ax.set_axis_off()
    gs = GridSpec(len(_metrics), len(_bevs))
    axes = [[None for _ in range(len(_metrics))] for _ in range(len(_bevs))]

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
                ax.plot(index, target_df.values,
                        label=target_name,
                        lw=1,
                        alpha=1.0 if highlight_target[target_name] else 0.8)
                loc = plticker.MaxNLocator(nbins=7, integer=True)  # this locator puts ticks at regular intervals
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
            else:
                [l.set_alpha(0.5) for l in ax.get_yticklabels()]


            #ax.legend()
            #
            # Last Metric Behaviour (Legend)
            if j == len(_metrics) - 1:
                ax.set_xlabel("Mission Time (min)")
                if i == 0:
                    ax.legend(loc="lower center",
                              bbox_to_anchor=(0, 0, 1, 1),
                              bbox_transform=fig.transFigure,
                              ncol=6, frameon=False, prop={'size':10})
                else:
                    pass
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

    fig.subplots_adjust(
            left=0.09, bottom=0.15, right=0.98, top=0.9, wspace=0.2, hspace=0.05)
    if title is not None and show_title:
        fig.suptitle(title, fontsize=12)
    if save_fig:
        fig.savefig(os.path.join(fig_path,title.replace(' ','_'))+'.png', transparent=True)
    else:
        fig.show()

    return fig

class SingleRunGraphing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(SingleRunGraphing, cls).setUpClass()
        #Grab results h5 file from set "good" results
        cls.fig_path = '/home/bolster/src/thesis/papers/active/16_MASS/figures'
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
            map_levels(cls.trust_observations, {'Waypoint': 'Control'})
            # Convert Sample Times into proper stuff
            cls.trust_observations.reset_index(inplace=True)
            cls.trust_observations['t']*=cls.trust_period
            cls.trust_observations.set_index(['var','run','observer','t','target'], inplace=True)

        data = cls.trust_observations.xs(cls.observer, level='observer').xs(cls.run_id, level='run')

        cls.deviance = pd.concat([gf - gf.mean() for g, gf in data.groupby(level=['var', 't'])])
        cls.sigmas = pd.concat([(gf / (gf.std(axis=0))).abs() for g, gf in cls.deviance.groupby(level=['var', 't'])])

    def test_position_track_plotting(self):
        fig = plot_axes_views_from_positions_frame(self.positions.xs('Waypoint', level='var').xs(self.run_id,level='run'))
        fig.savefig(os.path.join(self.fig_path, "Waypoint_Position_Track_Run_{0:d}.png".format(self.run_id)),
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
        fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 3.5), dpi=80)
        for i, metric in enumerate(self.sigmas.columns):
            ax=axes[i]
            summed_sigmas = self.sigmas.unstack('target')[metric].groupby(level='var').sum()

            summed_sigmas['Mean'] = summed_sigmas.mean(axis=1)
            _ = (summed_sigmas / offset-1).plot(ax=ax, kind='bar', legend=False)
            ax.grid(color='lightgray')
            ax.set_xlabel('Behaviour')
            ax.set_ylabel('Overall $\sigma$')
            ax.set_title(metric)
            for tick in ax.get_xticklabels():
                tick.set_rotation(0)
            ax.set_ylim([-1.0,1.0])
            fig.tight_layout()
            if metric == self.sigmas.columns[-1]:
                ax.legend(ncol=2)
        fig.savefig(os.path.join(self.fig_path, "summedsigmabar") + '.png', transparent=True)


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
        cls.fig_path = '/home/bolster/src/thesis/papers/active/16_MASS/figures'

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
            return classifier(gf, minority=True, _dixon_test=dixon_test_95)

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
            return classifier(gf, minority=False, _dixon_test=dixon_test_95)

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

def classifier(gf, minority=False, _dixon_test=dixon_test_95):
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
            print "Sneaky Contradictions!{}".format(outlier)
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
if __name__ == '__main__':
    unittest.main()
