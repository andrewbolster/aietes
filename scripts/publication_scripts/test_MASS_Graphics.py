import unittest
import os
import pandas as pd
from bounos.ChartBuilders import plot_axes_views_from_positions_frame
from matplotlib import pylab as plt
from matplotlib.gridspec import GridSpec


class smart_dict(dict):
    # Replaces missing lookups with the key
    def __missing__(self, key):
        return key

def plot_behaviour_metric_graph(data, fig_path, observer = 'Bravo', run = 0,
                                force_ymin=None, title=None, y_label_map=None,
                                save_fig=False):

    _bevs = set(data.index.get_level_values('var'))
    _metrics = list(data.columns)
    fig = plt.figure(figsize=(8, 5), dpi=80)
    base_ax = fig.add_axes([0, 0, 1, 1], )
    base_ax.set_axis_off()
    gs = GridSpec(len(_metrics), len(_bevs))
    axes = [[None for _ in range(len(_metrics))] for _ in range(len(_bevs))]
    # Column Behaviours
    for i, (behaviour, bev_df) in enumerate(data.groupby(level='var')):
        # Row Metrics
        metrics_df = bev_df
        if 'run' in bev_df.index.names:
            metrics_df = metrics_df.xs(run, level='run', drop_level=True)
        if 'observer' in bev_df.index.names:
            metrics_df = metrics_df.xs(observer, level='observer')
        for j, metric in enumerate(metrics_df):
            ax = fig.add_subplot(gs[j, i])
            for target, target_df in metrics_df[metric].groupby(level='target'):
                ax.plot(target_df.values, label=target,alpha=0.6)
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

            #ax.legend()
            #
            # Last Metric Behaviour (Legend)
            if j == len(_metrics) - 1:
                ax.set_xlabel("Time")
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
            left=0.09, bottom=0.12, right=0.98, top=0.9, wspace=0.2, hspace=0.05)
    if title is not None:
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
        cls.results_path = results_path_good_as_of_monday210416

        cls.run_id = 0
        cls.target = 'Alfa'
        cls.observer = 'Bravo'

        with pd.get_store(cls.results_path + '.h5') as store:
            cls.positions = store.positions
            cls.trust_observations = store.trust
            cls.trust_observations.columns.name = 'metric'

        data = cls.trust_observations.xs(cls.observer, level='observer').xs(cls.run_id, level='run')
        cls.deviance = pd.concat([gf - gf.mean() for g, gf in data.groupby(level=['var', 't'])])
        cls.sigmas = pd.concat([(gf / (gf.std(axis=0))).abs() for g, gf in cls.deviance.groupby(level=['var', 't'])])

    def test_position_track_plotting(self):
        fig = plot_axes_views_from_positions_frame(self.positions.xs('Waypoint', level='var'))
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
        for metric in self.sigmas.columns:
            summed_sigmas = self.sigmas.unstack('target')[metric].groupby(level='var').sum()

            summed_sigmas['Mean'] = summed_sigmas.mean(axis=1)
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8, 5), dpi=80)
            _ = (summed_sigmas - 180).plot(ax=ax, kind='bar', legend=False)
            ax.grid(color='lightgray')
            fig.tight_layout()
            title = "summedsigmabar_{}".format(metric)
            fig.savefig(os.path.join(self.fig_path, title.replace(' ', '_')) + '.png', transparent=True)

    def test_summed_sigma_confidence_table(self):
        summed_sigmas = self.sigmas.unstack('target').groupby(level='var').sum()
        intergfs = []
        for g, gf in summed_sigmas.stack('target').groupby(level='var'):
            _intergf = gf.unstack('target').stack('metric')
            intergfs.append(_intergf[self.target] / _intergf.drop(self.target, axis=1).mean(axis=1))
        with open(os.path.join(self.fig_path,"confidence.tex"), 'w') as f:
            pd.concat(intergfs).unstack('metric').to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))

if __name__ == '__main__':
    unittest.main()
