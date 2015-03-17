from  __future__ import division
import os
import pandas as pd
import numpy as np
import seaborn as sns
import quantities as q
from collections import OrderedDict
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
import itertools
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt

import aietes
import aietes.Tools as Tools
from aietes.Tools import map_levels
import bounos.ChartBuilders as cb
import bounos.Analyses.Trust as Trust
import bounos.multi_loader as multi_loader

# print(matplotlib.rcParams)
_boxplot_kwargs = {
    'showmeans': True,
    'showbox': False,
    'widths': 0.2,
    'linewidth': 2
}
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
golden_mean = (np.sqrt(5) - 1.0) / 2.5  # because it fits
w = 4

cb.latexify(columns=2, factor=0.45)

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing
#result_dirs_by_latest=sorted(filter(os.path.isdir,map(lambda p: os.path.abspath(os.path.join(Tools._results_dir,p)),os.listdir(Tools._results_dir))),key=lambda f:os.path.getmtime(f))
#last_bella_static_base=filter(lambda p: os.path.basename(p).startswith("CommsRateTest-bella_static"), result_dirs_by_latest)[-1]

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

scenario_map = dict(zip(
    [u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static', u'bella_static_median',u'bella_all_mobile_median'],
    ['All Mobile', '$n_1$ Static', '$n_1$ Mobile', 'All Static','Static using alternate filter', 'Mobile using alternate filter']
))
scenario_order = list(reversed([u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static']))

in_results = partial(os.path.join, Tools._results_dir)


def per_scenario_gd_mal_trusts(gd_file, mal_file):
    if not all(map(os.path.isfile, [gd_file, mal_file])):
        raise OSError("Either {} or {} is not present".format(gd_file, mal_file))
    with pd.get_store(mal_file) as store:
        mal_trust = store.get('trust')
        map_levels(mal_trust, scenario_map)
    with pd.get_store(gd_file) as store:
        gd_trust = store.get('trust')
        map_levels(gd_trust, scenario_map)
    return gd_trust, mal_trust


def plot_comparison(gd_tp, mal_tp, s, trust="grey_", metric=None, show_title=True, keyword=None):
    tex_safe_s = scenario_map[s]

    df1 = Trust.generate_mtfm(gd_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)
    df2 = Trust.generate_mtfm(mal_tp, 'n0', 'n1', ['n2', 'n3'], ['n4', 'n5']).sum(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(w, w * golden_mean), sharey=True)

    x = df1.index.levels[df1.index.names.index('t')]

    ax.plot(x, df1, label="Fair", linestyle="--")
    ax.plot(x, df2, label="Selfish")

    top = df1.mean() + df1.std()
    middle = df1.mean()
    bottom = df1.mean() - df1.std()

    for line in [top, middle, bottom]:
        ax.axhline(line, linestyle=':', alpha=0.5)

    ax.fill_between(x, df2, bottom, where=df2 < bottom, interpolate=True, facecolor='green', alpha=0.25)
    ax.fill_between(x, df2, top, where=df2 > top, interpolate=True, facecolor='green', alpha=0.25)

    if show_title:
        ax.set_title("{} {}".format(tex_safe_s, "Emphasising {}".format(metric) if metric else ""))
    ax.set_ylim([0, 1])
    if df2.mean() > 0.5: # Selfish bar is much more highlighted
        ax.legend(loc='lower left', ncol=2)
    else:
        ax.legend(loc='upper left', ncol=2)

    ax.axhline(0.5, linestyle="..")
    ax.set_ylabel('{}Trust Value'.format(trust.replace("_", " ").title()))
    ax.set_xlabel('Trust Assessment Period')
    fig.tight_layout(pad=0.1)
    fig.savefig("/home/bolster/src/thesis/papers/active/15_AdHocNow/img/trust_{}_{}{}.pdf".format(
        s, "emph_%s" % metric if metric else "even",
        "_%s" % keyword if keyword else ""),transparent = True
    )


def cross_scenario_plot():
    gd_tp, mal_tp = per_scenario_gd_mal_trusts()
    with mpl.rc_context(rc={'text.usetex': 'True'}):
        for s in scenario_order:
            plot_comparison(gd_tp, mal_tp, s, metric="All_")


def plot_weight_comparisons(gd_file, mal_file, malicious_behaviour="Selfish", s="bella_static", excluded=[],
                            show_title=True):
    with mpl.rc_context(rc={'text.usetex': 'True'}):
        gd_trust, mal_trust = per_scenario_gd_mal_trusts(gd_file, mal_file)
        print gd_trust.keys()
        gd_tp, mal_tp = per_scenario_gd_mal_trust_perspective(gd_trust, mal_trust, s=s)
        plot_comparison(gd_tp, mal_tp, s, show_title=show_title, keyword=malicious_behaviour)
        for i, mi in enumerate(trust_metrics):
            if mi not in excluded:
                gd_tp, mal_tp = per_scenario_gd_mal_trust_perspective(gd_trust, mal_trust, s=s,
                                                                      weight_vector=weight_for_metric(mi, 3))
                plot_comparison(gd_tp, mal_tp, s, metric=mi, show_title=show_title,
                                keyword=malicious_behaviour)


def per_scenario_gd_mal_trust_perspective(gd_trust, mal_trust, weight_vector=None, s=None, two_pass=False):
    if weight_vector is not None:
        print("Using {}".format(weight_vector))
    if s is not None:
        print("Trimming trust to {}".format(s))
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
        trust_type = "{}_".format(trust_type)
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
    ax[0].set_ylabel('{}Trust Assessment'.format(trust_type.replace("_", " ").title()))
    ax[1].set_ylabel('{}Trust Assessment'.format(trust_type.replace("_", " ").title()))

    fig.tight_layout()
    fig.savefig("/home/bolster/src/thesis/papers/active/15_AdHocNow/img/{}trust_{}_joint.pdf".format(
        trust_type, s), transparent=True
    )


def weight_for_metric(m, emph=4):
    base = np.ones_like(trust_metrics, dtype=np.float)
    base[np.where(trust_metrics == m)] += emph
    return pd.Series(base / base.sum(), index=trust_metrics)


def beta_trusts(trust, length=4096):
    trust['+'] = (trust.TXThroughput / length) * (1 - trust.PLR)
    trust['-'] = (trust.TXThroughput / length) * (trust.PLR)
    beta_trust = trust[['+', '-']].unstack(level='target')
    return beta_trust


def beta_otmf_vals(beta_trust):
    beta_t_confidence = lambda s, f: 1 - np.sqrt((12 * s * f) / ((s + f + 1) * (s + f) ** 2))
    beta_t = lambda s, f: s / (s + f)
    otmf_T = lambda s, f: 1 - np.sqrt(
        ((((beta_t(s, f) - 1) ** 2) / 2) + (((beta_t_confidence(s, f) - 1) ** 2) / 9))) / np.sqrt((1 / 2) + (1 / 9))
    beta_vals = beta_trust.apply(lambda r: beta_t(r['+'],r['-']), axis=1)
    otmf_vals = beta_trust.apply(lambda r: otmf_T(r['+'],r['-']), axis=1)
    return beta_vals, otmf_vals

def plot_mtfm_boxplot(filename, s=None, show_title=False, keyword=None):
    with mpl.rc_context(rc={'text.usetex':'True'}):
        with pd.get_store(in_results(filename)) as store:
            trust=store.get('trust')
            tp_net=Trust.network_trust_dict(Trust.generate_node_trust_perspective(trust))
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
                title = "Selfish {}".format(tex_safe_s)
                fig = cb.trust_network_wrt_observers(tp_net.xs(tex_safe_s, level='var'),
                                                     tex_safe_s, title=show_title,
                                                     figsize=(w,w*golden_mean))
                #fig.tight_layout(pad=0.1)
                print tex_safe_s
                fig.savefig("/home/bolster/src/thesis/papers/active/15_AdHocNow/img/trust_{}{}.pdf".format(
                    s, "_"+keyword if keyword is not None else ""),
                            transparent=True)

def run_malicious_comparison(args):
    if args.scenario is not None:
        if isinstance(args.scenario, list):
            scenarios = args.scenario
        else:
            scenarios = [args.scenario]
    else:
        scenarios = scenario_order
    for s in scenarios:
        print s
        plot_weight_comparisons(in_results(args.good_file), in_results(args.bad_file),
                                s=s,
                                malicious_behaviour=args.bad_name,
                                show_title=False)


def plot_triplet(args):
    gd_trust, mal_trust = per_scenario_gd_mal_trusts(in_results(good), in_results(malicious))
    print gd_trust.keys()
    mal_mobile = mal_trust.xs('All Mobile', level='var')
    gd_mobile = gd_trust.xs('All Mobile', level='var')

    gd_beta_t = beta_trusts(gd_mobile)
    mal_beta_t = beta_trusts(mal_mobile)

    fig, ax = plt.subplots(1, 1, figsize=(w, w * golden_mean), sharey=True)

    x = df1.index.levels[df1.index.names.index('t')]

    ax.plot(x, df1, label="Fair", linestyle="--")
    ax.plot(x, df2, label="Selfish")

    top = df1.mean() + df1.std()
    middle = df1.mean()
    bottom = df1.mean() - df1.std()

    for line in [top, middle, bottom]:
        ax.axhline(line, linestyle=':', alpha=0.5)

    ax.fill_between(x, df2, bottom, where=df2 < bottom, interpolate=True, facecolor='green', alpha=0.25)
    ax.fill_between(x, df2, top, where=df2 > top, interpolate=True, facecolor='green', alpha=0.25)

    if show_title:
        ax.set_title("{} {}".format(tex_safe_s, "Emphasising {}".format(metric) if metric else ""))
    ax.set_ylim([0, 1])
    if df2.mean() > 0.5:  # Selfish bar is much more highlighted
        ax.legend(loc='lower left', ncol=2)
    else:
        ax.legend(loc='upper left', ncol=2)

    ax.axhline(0.5, linestyle="..")
    ax.set_ylabel('{}Trust Value'.format(trust.replace("_", " ").title()))
    ax.set_xlabel('Trust Assessment Period')
    fig.tight_layout(pad=0.1)
    fig.savefig("/home/bolster/src/thesis/papers/active/15_AdHocNow/img/trust_{}_{}{}.pdf".format(
        s, "emph_%s" % metric if metric else "even",
        "_%s" % keyword if keyword else ""), transparent=True
    )

if __name__ == "__main__":
    good = "TrustMobilityTests-0.015-4-2015-02-19-09-11-25.h5"
    malicious = "MaliciousBadMouthingPowerControlTrustMobilityTests-0.015-4-2015-02-19-09-08-49.h5"
    good = "TrustMobilityTests-0.025-3-2015-02-19-10-53-28.h5"
    malicious = "MaliciousBadMouthingPowerControlTrustMobilityTests-0.025-3-2015-02-19-10-50-43.h5"
    #malicious = "MaliciousBadMouthingPowerControlTrustMedianTests-0.025-3-2015-02-19-23-27-01.h5"
    #good = "TrustMedianTests-0.025-3-2015-02-19-23-29-39.h5"

    import argparse
    parser = argparse.ArgumentParser(description="Compare Weights across Runs")
    parser.add_argument('--good_file', type=str, help="Good Behaviour File",
                        default=good)
    parser.add_argument('--bad_file', type=str, help="Bad Behaviour File",
                        default=malicious)
    parser.add_argument('--bad_name', type=str, help="Bad Behaviour Name",
                        default="BadMouthingPowerControl")
    parser.add_argument('--scenario', type=str, help="Selected Scenario",
                        default="bella_static")
    parser.add_argument('--keyword', type=str, help="Tag to go on the end of MTFM filenames",
                        default=None)

    parser.add_argument('--wcomparison', action='store_true', help="Run Weight Comparisons on a Malicious and Good Set",
                        default=False)
    parser.add_argument('--mtfmboxplot', action='store_true', help="Run MTMF comparison on a set",
                        default=False)
    parser.add_argument('--betaplot', action='store_true', help="Run Beta Comparison on a set",
                        default=False)


    args = parser.parse_args()
    if args.scenario is None:
        args.scenario = ["bella_all_mobile"]
    if args.wcomparison:
        run_malicious_comparison(args)
    elif args.mtfmboxplot:
        plot_mtfm_boxplot(good, s=args.scenario, keyword=args.keyword)
    elif args.betaplot:
        plot_triplet(args)
        parser.print_help()
