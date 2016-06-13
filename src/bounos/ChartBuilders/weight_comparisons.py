# coding=utf-8
from __future__ import division
import os
import itertools
import warnings

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import aietes.Tools as Tools
from aietes.Tools import map_levels
from bounos.ChartBuilders import format_axes, latexify, trust_network_wrt_observers
import bounos.Analyses.Trust as Trust
from bounos.Analyses import scenario_order, scenario_map

# print(matplotlib.rcParams)
_boxplot_kwargs = {
    'showmeans': True,
    'showbox': False,
    'widths': 0.2,
    'linewidth': 2
}
golden_mean = (np.sqrt(5) - 1.0) / 2.5  # because it fits
w = 4

latexify(columns=2, factor=0.45)

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing
# result_dirs_by_latest=sorted(filter(os.path.isdir,map(lambda p: os.path.abspath(os.path.join(Tools._results_dir,p)),os.listdir(Tools._results_dir))),key=lambda f:os.path.getmtime(f))
# last_bella_static_base=filter(lambda p: os.path.basename(p).startswith("CommsRateTest-bella_static"), result_dirs_by_latest)[-1]

trust_metrics = np.asarray("ADelay,ARXP,ATXP,RXThroughput,PLR,TXThroughput".split(','))

exclude = []

trust_combinations = []
map(trust_combinations.extend,
    np.asarray([itertools.combinations(trust_metrics, i)
                for i in range(5, len(trust_metrics))])
    )

trust_combinations = np.asarray(filter(lambda x: all(map(lambda m: m not in exclude, x)), trust_combinations))
trust_metric_selections = np.asarray(
    [map(lambda m: float(m in trust_combination), trust_metrics) for trust_combination in trust_combinations])
trust_metric_weights = map(lambda s: s / sum(s), trust_metric_selections)


def per_scenario_gd_mal_trusts(gd_file, mal_file):
    if not all(map(os.path.isfile, [gd_file, mal_file])):
        if all(map(os.path.isfile, map(Tools.in_results, [gd_file, mal_file]))):
            gd_file, mal_file = map(Tools.in_results, [gd_file, mal_file])
        else:
            raise OSError("Either {0} or {1} is not present".format(gd_file, mal_file))
    with pd.get_store(mal_file) as store:
        mal_trust = store.get('trust')
        map_levels(mal_trust, scenario_map)
    with pd.get_store(gd_file) as store:
        gd_trust = store.get('trust')
        map_levels(gd_trust, scenario_map)
    return gd_trust, mal_trust


def plot_comparison(df1, df2, s, trust="grey_", metric=None, show_title=True, keyword=None,
                    figsize=None, labels=None, prefix="img/", extension="pdf", show_grid=True):
    if labels is None:
        labels = ["Fair", "Selfish"]

    tex_safe_s = scenario_map[s]

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
    ax.grid(show_grid)

    x = df1.index.levels[df1.index.names.index('t')]

    ax.plot(x, df1, label=labels[0])
    ax.plot(x, df2, label=labels[1])

    top = df1.mean() + df1.std()
    middle = df1.mean()
    bottom = df1.mean() - df1.std()

    for line in [top, middle, bottom]:
        ax.axhline(line, linestyle=':', alpha=0.2)

    ax.fill_between(x, df2, bottom, where=df2 < bottom, interpolate=True, facecolor='green', alpha=0.25)
    ax.fill_between(x, df2, top, where=df2 > top, interpolate=True, facecolor='green', alpha=0.25)

    if show_title:
        ax.set_title("{0} {1}".format(tex_safe_s, "Emphasising {0}".format(metric) if metric else ""))
    ax.set_ylim([0, 1])
    if df2.mean() > 0.5:  # Selfish bar is much more highlighted
        ax.legend(loc='lower left', ncol=2)
    else:
        ax.legend(loc='upper left', ncol=2)

    ax.set_ylabel('{0}Trust Value'.format(trust.replace("_", " ").title()))
    ax.set_xlabel('Mission Time (mins)')
    ax.xaxis.set_major_formatter(plticker.FuncFormatter(lambda x,pos: int(x*10)))
    ax = format_axes(ax, show_grid=show_grid)
    fig.tight_layout(pad=0.1)
    fig.savefig("{0}trust_{1}_{2}{3}.{4}".format(
        prefix,
        s, "emph_{0!s}".format(metric) if metric else "even",
        "_{0!s}".format(keyword) if keyword else "",
        extension), transparent=True
    )
    plt.close(fig)


def cross_scenario_plot():
    gd_tp, mal_tp = per_scenario_gd_mal_trusts()
    with mpl.rc_context(rc={'text.usetex': 'True'}):
        for s in scenario_order:
            plot_comparison(gd_tp, mal_tp, s, metric="All_")


def plot_weight_comparisons(gd_file, mal_file,
                            malicious_behaviour="Selfish", s="bella_static",
                            excluded=None, show_title=True, figsize=None,
                            labels=None, prefix="img/", extension="pdf"):
    if labels is None:
        labels = ["Fair", "Selfish"]

    if excluded is None:
        excluded = []

    mtfm_args = ('n0', 'n1', ['n2', 'n3'], ['n4', 'n5'])
    with mpl.rc_context(rc={'text.usetex': 'True'}):

        gd_trust, mal_trust = per_scenario_gd_mal_trusts(gd_file, mal_file)
        gd_tp, mal_tp = per_scenario_gd_mal_trust_perspective(gd_trust, mal_trust, s=s)

        gd_mtfm = Trust.generate_mtfm(gd_tp, *mtfm_args).sum(axis=1)
        mal_mtfm = Trust.generate_mtfm(mal_tp, *mtfm_args).sum(axis=1)

        plot_comparison(gd_mtfm, mal_mtfm, s=s,
                        show_title=show_title, keyword=malicious_behaviour,
                        figsize=figsize, labels=labels, prefix=prefix, extension=extension, show_grid=False)
        for i, mi in enumerate(trust_metrics):
            if mi not in excluded:
                gd_tp, mal_tp = per_scenario_gd_mal_trust_perspective(gd_trust, mal_trust, s=s,
                                                                      weight_vector=weight_for_metric(mi, 3))

                gd_mtfm = Trust.generate_mtfm(gd_tp, *mtfm_args).sum(axis=1)
                mal_mtfm = Trust.generate_mtfm(mal_tp, *mtfm_args).sum(axis=1)
                plot_comparison(gd_mtfm, mal_mtfm, s, metric=mi, show_title=show_title, prefix=prefix,
                                keyword=malicious_behaviour, figsize=figsize, labels=labels,
                                extension=extension, show_grid=False)


def per_scenario_gd_mal_trust_perspective(gd_trust, mal_trust, weight_vector=None, s=None,
                                          two_pass=False):
    # TODO This is useless, refactor
    if weight_vector is not None:
        print("Using {0}".format(weight_vector.values))
    if s is not None:
        print("Trimming trust to {0}".format(s))
        mal_trust = mal_trust.xs(scenario_map[s], level='var')
        gd_trust = gd_trust.xs(scenario_map[s], level='var')
    mal_tp = Trust.generate_node_trust_perspective(mal_trust, metric_weights=weight_vector)
    map_levels(mal_tp, scenario_map)
    gd_tp = Trust.generate_node_trust_perspective(gd_trust, metric_weights=weight_vector)
    map_levels(gd_tp, scenario_map)

    return gd_tp, mal_tp


def plot_scenario_pair(gd_tp, mal_tp, s, span=1, trust_type=None):
    if trust_type is None:
        trust_type = ""
    else:
        trust_type = "{0}_".format(trust_type)
    tex_safe_s = scenario_map[s]
    df1 = pd.stats.moments.ewma(gd_tp.xs(tex_safe_s, level='var'), span=span).drop('n1', level='observer')
    df2 = pd.stats.moments.ewma(mal_tp.xs(tex_safe_s, level='var'), span=span).drop('n1', level='observer')
    fig, ax = plt.subplots(1, 2, figsize=(2 * w, w * golden_mean), sharey=True)
    df1['n1'].xs('n0', level='observer').plot(ax=ax[0], use_index=False, legend=True, )
    df2['n1'].xs('n0', level='observer').plot(ax=ax[1], use_index=False)
    ax[0].set_title("All Fair")
    ax[1].set_title("Malicious $n_1$")
    ax[0].set_ylim([0, 1])
    ax[0].axhline(0.5, linestyle="dashed")
    ax[1].axhline(0.5, linestyle="dashed")
    ax[0].set_ylabel('{0}Trust Assessment'.format(trust_type.replace("_", " ").title()))
    ax[1].set_ylabel('{0}Trust Assessment'.format(trust_type.replace("_", " ").title()))

    fig.tight_layout()
    fig.savefig("img/{0}trust_{1}_joint.pdf".format(
        trust_type, s), transparent=True
    )


def weight_for_metric(m, emph=4):
    base = np.ones_like(trust_metrics, dtype=np.float)
    if isinstance(m, list):
        for _ in m:
            base[np.where(trust_metrics == m)] += emph
    else:
        base[np.where(trust_metrics == m)] += emph
    return norm_weight(base)


def norm_weight(base, metric_names=None):
    if isinstance(base, pd.Series):
        normed = pd.Series(base / base.abs().sum())
    else:
        normed = pd.Series(base, index=metric_names) / sum(map(abs, base))

    return normed


def beta_trusts(trust, length=4096):
    trust['+'] = (trust.TXThroughput / length) * (1 - trust.PLR)
    trust['-'] = (trust.TXThroughput / length) * trust.PLR
    beta_trust = trust[['+', '-']].unstack(level='target')
    return beta_trust


def beta_otmf_vals(beta_trust):
    beta_t_confidence = lambda s, f: 1 - np.sqrt((12 * s * f) / ((s + f + 1) * (s + f) ** 2))
    beta_t = lambda s, f: s / (s + f)
    otmf_t = lambda s, f: 1 - np.sqrt(
        ((((beta_t(s, f) - 1) ** 2) / 2) + (((beta_t_confidence(s, f) - 1) ** 2) / 9))) / np.sqrt((1 / 2) + (1 / 9))
    beta_vals = beta_trust.apply(lambda r: beta_t(r['+'], r['-']), axis=1)
    otmf_vals = beta_trust.apply(lambda r: otmf_t(r['+'], r['-']), axis=1)
    return beta_vals, otmf_vals


def plot_mtfm_boxplot(filename, s=None, show_title=False, keyword=None, prefix="img/",
                      metric_weights=None, figsize=None, xlabel=True, dropnet=False,
                      extension="pdf"):
    if figsize is None:
        figsize = (w, w * golden_mean)
    with mpl.rc_context(rc={'text.usetex': 'True'}):
        with pd.get_store(Tools.in_results(filename)) as store:
            trust = store.get('trust')
            tp_net = Trust.network_trust_dict(
                Trust.generate_node_trust_perspective(trust, par=True,
                                                      metric_weights=metric_weights)
            )
            map_levels(tp_net, scenario_map)
            if s is not None:
                if isinstance(s, list):
                    scenarios = s
                else:
                    scenarios = [s]
            else:
                scenarios = scenario_order
            for s in scenarios:
                tex_safe_s = scenario_map[s]
                title = "{0}{1}".format(keyword, tex_safe_s)
                try:
                    fig = trust_network_wrt_observers(tp_net.xs(tex_safe_s, level='var'),
                                                      tex_safe_s, title=title if show_title else False,
                                                      figsize=figsize, xlabel=xlabel, dropnet=dropnet)
                    fig.tight_layout(pad=0)
                    fig.savefig("{0}trust_{1}{2}.{3}".format(
                        prefix,
                        s,
                        "_" + keyword if keyword is not None else "",
                        extension),
                        transparent=True)
                except KeyError:
                    warnings.warn("Scenario {0} not in trust run, skipping".format(tex_safe_s))
