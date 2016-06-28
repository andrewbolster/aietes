# coding=utf-8
import itertools
import os
from collections import OrderedDict, defaultdict
import warnings

import functools
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import pylatex
from matplotlib import pylab as plt, cm, cm, cm, cm, gridspec
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import interpolate as interpolate
import matplotlib2tikz as mpl2tkz

from aietes import Tools
from bounos.Analyses import Trust
from bounos.ChartBuilders import latexify, plot_nodes, format_axes, unique_cm_dict_from_list, _texcol, _texcolhalf, _texcolthird, _texfac

in_results = functools.partial(os.path.join, Tools._results_dir)
print Tools._results_dir
app_rate_from_path = lambda s: float(".".join(s.split('-')[2].split('.')[0:-1]))
scenario_map = dict(zip(
    [u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static'],
    ['All Mobile', '$n_1$ Static', '$n_1$ Mobile', 'All Static']
))
scenario_order = list(reversed([u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static']))

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6

latexify(columns=_texcol, factor=_texfac)

# # Data File Acquistion

# In[4]:

_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing
result_h5s_by_latest = sorted(filter(lambda p: os.path.basename(p).endswith("h5"),
                                     map(lambda p: os.path.abspath(os.path.join(Tools._results_dir, p)),
                                         os.listdir(Tools._results_dir))), key=lambda f: os.path.getmtime(f))
rate_and_ranges = filter(lambda p: os.path.basename(p).startswith("CommsRateAndRangeTest"),
                         result_h5s_by_latest)
app_rates = map(app_rate_from_path, rate_and_ranges)


def interpolate_rate_sep(df, key):
    x, y, z = df.rate, df.separation, df[key]

    xi = np.linspace(x.min(), x.max(), 16)
    yi = np.linspace(y.min(), y.max(), 16)
    # VERY IMPORTANT, to tell matplotlib how is your data organized
    zi = interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    return xi, yi, zi, x, y


def plot_lines_of_throughput(df):
    fig = plt.figure(figsize=(w, golden_mean * w), facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    for (k, g), ls in zip(df.groupby('separation'), itertools.cycle(["-", "--", "-.", ":"])):
        ax.plot(g.rate, g.throughput, label=k, linestyle=ls)
    ax.legend(loc="upper left")
    ax.set_xlabel("Packet Emission Rate (pps)")
    ax.set_ylabel("Per Node Avg. Throughput (bps)")

    return fig


def plot_contour_pair(xi, yi, zi):
    fig = plt.figure(figsize=(2 * w, golden_mean * w * 2))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    cs = plt.contour(xi, yi, zi, 15, linewidths=0.5, color='k')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    xig, yig = np.meshgrid(xi, yi)

    surf = ax.plot_surface(xig, yig, zi, linewidth=0)
    return fig


def plot_contour_3d(xi, yi, zi, rot=120, labels=None):
    fig = plt.figure(figsize=(w, golden_mean * w))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # Normalise Z
    zi_norm = zi / np.nanmax(zi)
    xig, yig = np.meshgrid(xi, yi)

    if rot < 90:
        zoffset = np.nanmin(zi)
        xoffset = np.nanmin(xig)
        yoffset = np.nanmin(yig)
    elif rot < 180:
        zoffset = np.nanmin(zi)
        xoffset = np.nanmax(xig)
        yoffset = np.nanmin(yig)
    else:
        zoffset = np.nanmin(zi)
        xoffset = np.nanmax(xig)
        yoffset = np.nanmax(yig)

    ax.plot_surface(xig, yig, zi, rstride=1, cstride=1, alpha=0.45, facecolors=cm.coolwarm(zi_norm), linewidth=1,
                    antialiased=True)
    cset = ax.contour(xig, yig, zi, zdir='z', offset=zoffset, linestyles='dashed', cmap=cm.coolwarm)
    cset = ax.contour(xig, yig, zi, zdir='x', offset=xoffset, cmap=cm.coolwarm)
    cset = ax.contour(xig, yig, zi, zdir='y', offset=yoffset, cmap=cm.coolwarm)
    ax.view_init(30, rot)

    if labels is not None:
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'])
        ax.set_zlabel(labels['z'])

    fig.tight_layout()
    return fig


def plot_contour_2d(xi, yi, zi, x=None, y=None, var=None, norm=False, figsize=None):
    if x is None:
        x = []
    if y is None:
        y = []

    if figsize is None:
        figsize = (w, golden_mean * w)

    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = fig.add_subplot(1, 1, 1)
    xig, yig = np.meshgrid(xi, yi)
    x_min, y_min = map(np.nanmin, (xi, yi))
    x_max, y_max = map(np.nanmax, (xi, yi))
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([x_min, x_max])
    ax.set_xlabel("Packet Emission Rate (pps)")
    ax.set_ylabel("Average Node Separation (m)")

    if norm:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(abs(zi))
        vmax = np.nanmax(abs(zi))

    cset = ax.contourf(zi, alpha=0.75,  # hatches=['+','x','-', '/', '\\', '//'],
                       cmap=plt.get_cmap('hot_r'),
                       vmin=vmin, vmax=vmax,
                       extent=[x_min, x_max, y_min, y_max]
                       )
    cbar = fig.colorbar(cset)
    if var is not None:
        # ax.set_title("{} with varying Packet Emission and Node Separations".format(var))
        cbar.set_label(var)

    if len(x) and len(y):
        ax.scatter(x, y, color='k', marker='.', s=1)

    # ax.clabel(cset,inline=1)
    return fig


def plot_all_funky_stuff(sdf, mobility):
    xt, yt, zt, x, y = interpolate_rate_sep(sdf.dropna(), "throughput")
    fig = plot_contour_2d(xt, yt, zt, x, y, "Per Node Avg. Throughput (bps)")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/throughput_2d_{0}.pdf".format(mobility))

    xd, yd, zd, x, y = interpolate_rate_sep(sdf.dropna(), "average_rx_delay")
    fig = plot_contour_2d(xd, yd, zd, x, y, "Average Delay (s)")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/delay_2d_{0}.pdf".format(mobility))

    xd, yd, zd, x, y = interpolate_rate_sep(sdf, "tdivdel")
    fig = plot_contour_2d(xd, yd, zd, x, y, "Throughput Delay Ratio")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/2d_ratio_{0}.pdf".format(mobility))

    fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
    fig.tight_layout(pad=0.1)
    fig.savefig("img/3d_ratio_{0}.pdf".format(mobility), transparent=True, facecolor='white')

    xd, yd, zd, x, y = interpolate_rate_sep(sdf, "co_norm")
    fig = plot_contour_2d(xd, yd, zd, x, y, "Normalised Throughput Delay Product", norm=True)
    fig.tight_layout(pad=0.1)
    fig.savefig("img/2d_normed_product_{0}.pdf".format(mobility))

    fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
    fig.tight_layout(pad=0.1)
    fig.savefig("img/3d_normed_product_{0}.pdf".format(mobility), transparent=True, facecolor='white')

    fig = plot_lines_of_throughput(sdf)
    fig.tight_layout(pad=0.1)
    fig.savefig("img/throughput_sep_lines_{0}.pdf".format(mobility), transparent=True, facecolor='white')


def get_mobility_stats(mobility):
    statsd = OrderedDict()
    trustd = OrderedDict()
    norm = lambda df: (df - np.nanmin(df)) / (np.nanmax(df) - np.nanmin(df))
    rate_and_ranges = filter(
        lambda p: os.path.basename(p).startswith("CommsRateAndRangeTest-bella_{0}".format(mobility)),
        result_h5s_by_latest)
    if not rate_and_ranges:
        raise ValueError("No Entries with mobility {0}".format(mobility))
    for store_path in sorted(rate_and_ranges):
        with pd.get_store(store_path) as s:
            stats = s.get('stats')
            trust = s.get('trust')
            # Reset Range for packet emission rate
            stats.index = stats.index.set_levels([
                                                     np.int32(
                                                         (np.asarray(stats.index.levels[0].astype(np.float64)) * 100)),
                                                     # Var
                                                     stats.index.levels[1].astype(np.int32)  # Run
                                                 ] + (stats.index.levels[2:])
                                                 )

            statsd[app_rate_from_path(store_path)] = stats.copy()
            trustd[app_rate_from_path(store_path)] = trust.copy()

    def df_for_rates_and_sep(d, last_names):
        df = pd.concat(d.values(), keys=d.keys(), names=['rate', 'separation'] + last_names[1:])

        return df

    sdf = df_for_rates_and_sep(statsd, stats.index.names)
    tdf = df_for_rates_and_sep(trustd, trust.index.names)
    sdf['throughput'] = sdf.throughput / 3600.0
    sdf = sdf.groupby(level=['rate', 'separation']).mean().reset_index()

    sdf['average_rx_delay_norm'] = 1 - norm(sdf.average_rx_delay)
    sdf['throughput_norm'] = norm(sdf.throughput)
    sdf['co_norm'] = sdf.average_rx_delay_norm * sdf.throughput_norm
    sdf = sdf.set_index(['rate', 'separation'])
    sdf['tdivdel'] = sdf.throughput / sdf.average_rx_delay
    sdf.reset_index(inplace=True)
    return sdf


def app_rate_from_path(s):
    return float(".".join(s.split('-')[2].split('.')[0:-1]))


result_h5s_by_latest = sorted(
    filter(
        lambda p: os.path.basename(p).endswith("h5"),
        map(lambda p: os.path.abspath(os.path.join(Tools._results_dir, p)),
            os.listdir(Tools._results_dir))
    ), key=lambda f: os.path.getmtime(f)
)
rate_and_ranges = filter(
    lambda p: os.path.basename(p).startswith("CommsRateAndRangeTest"),
    result_h5s_by_latest
)


def get_emmission_stats(df, separation=100):
    stats = df.swaplevel('rate', 'separation').xs(separation, level='separation')
    stats.index.names = ['var'] + stats.index.names[1:]

    # Reset Range for packet emission rate
    stats.index = stats.index.set_levels([
                                             stats.index.levels[0].astype(np.float64),  # Var
                                             stats.index.levels[1].astype(np.int32)  # Run
                                         ] + (stats.index.levels[2:])
                                         )
    return stats


def get_separation_stats(df, emission=0.015):
    stats = df.xs(emission, level='rate')
    stats.index.names = ['var'] + stats.index.names[1:]

    # Reset Range for packet emission rate
    stats.index = stats.index.set_levels([
                                             stats.index.levels[0].astype(np.int32),  # Var
                                             stats.index.levels[1].astype(np.int32)  # Run
                                         ] + (stats.index.levels[2:])
                                         )
    return stats


def interpolate_rate_sep(df, key, method='linear'):
    X, Y, Z = df.rate, df.separation, df[key]

    xi = np.linspace(X.min(), X.max(), 16)
    yi = np.linspace(Y.min(), Y.max(), 16)
    # VERY IMPORTANT, to tell matplotlib how is your data organized
    zi = interpolate.griddata(points=(X, Y), values=Z, xi=(xi[None, :], yi[:, None]), method=method)
    return xi, yi, zi, X, Y


def savefig(fig, name, extn="pdf", tight=True, ax=None, **kwargs):
    _kwargs = Tools.kwarger()
    _kwargs.update(kwargs)
    if tight:
        fig.tight_layout(pad=0.1)
    if ax is not None:
        if isinstance(ax, list):
            map(format_axes, ax)
        else:
            format_axes(ax)
    fig.savefig("{0}.{1}".format(name, extn), **_kwargs)
    try:
        mpl2tkz.save("{0}.tex".format(name), fig, show_info=False)
    except:
        warnings.warn("Couldn't tkzify {0}, skipping".format(name))
    plt.close(fig)


def saveinput(text, name, extn='tex'):
    Tools.mkdir_p('input')
    with open("input/{0}.{1}".format(name, extn), 'w') as f:
        f.write(text)


def generate_figure_contact_tex(fig_paths, target_path='.'):
    """
    Generate a contact sheet of the listed figure paths.
    Assumes figpaths are graphics files acceptable to pdflatex
    :param fig_paths: iterable of image paths
    :param target_path: destination directory
    :return:
    """
    doc = pylatex.Document(default_filepath=os.path.join(target_path, 'generated_figures'))

    with doc.create(pylatex.Section('Generated Figures')):
        for image_path in sorted(fig_paths):
            filename = os.path.split(image_path)[-1]
            with doc.create(pylatex.Figure(position='h!')) as fig:
                fig.add_image(image_path, width=pylatex.NoEscape(r'\linewidth'))
                fig.add_caption(filename)

    doc.generate_tex()


phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']
key_order = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR', 'INDD', 'INHD', 'Speed']
comm_keys_alt = ['ATXP', 'RXThroughput', 'TXThroughput', 'PLR', 'INDD']
phys_keys_alt = ['ADelay', 'ARXP', 'INDD', 'INHD', 'Speed']
observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9
results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
fig_basedir = "/home/bolster/src/thesis/Figures"


def subset_renamer(s):
    """
    Capitalise a string and make it tex-safe (i.e. avoid the madness of _alt)
    :param s:
    :return: string
    """

    s = s.replace("_alt", " Alt.").replace("signed", "").replace("only_feats", "_").replace("_"," ")
    s = str.capitalize(s)
    return s


def plot_trust_line_graph(result, title=None, stds=True, spans=None, box=None, means=None, target=None,
                          _texcol=_texcol, _texfac=_texfac, plot_filename=None, palette=None):
    """
    Plot weighted trust result line graph (i.e. T vs t)
    Optionally ewma smooth the results with `spans`>0
    Optionally plot +- std-envelop around means.

    By default, mean lines are across time.

    :param result:
    :param title:
    :param stds:
    :param spans:
    :param box:
    :param means: ['time','instantaneous']
    :param target: column name in results that maps to the "target" or "suspect" node. if none given, assumes first column
    :param plot_filename: if given, only return the intermediate results and plot using the given filename
    :return:
    """

    _target_alpha = 1.0
    _default_alpha = 0.6

    if means is None:
        means = 'time'
    if means not in ['time', 'instantaneous']:
        raise ValueError("Invalid `means` argument")

    if target is None:
        target = result.columns[0]

    target_index = result.columns.get_loc(target)

    dropna = lambda x: x[~np.isnan(x)]

    fig_size = latexify(columns=_texcol, factor=_texfac)

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax = plt.subplot(gs[0])
    axb = None
    if spans is not None:
        plottable = pd.stats.moments.ewma(result, span=spans)
    else:
        plottable = result

    def pltstd(tup):
        mean, std = tup
        ax.axhline(mean + std, alpha=0.3, ls=':', color='green')
        ax.axhline(mean - std, alpha=0.3, ls=':', color='red')

    lines = {}
    if palette is None:
        palette = unique_cm_dict_from_list(plottable.columns.tolist())
    if means is 'time':
        # Plot all lines
        for c in plottable.columns:
            if c == target:
                lines[target] = plottable[target].plot(ax=ax, alpha=_target_alpha, lw=1, color=palette[c])
            else:
                lines[c] = plottable[c].plot(ax=ax, alpha=_default_alpha, lw=1, color=palette[c])
        if box is None:
            for k, v in result.mean().iteritems():
                if k != target:  # Everyone Else
                    ax.axhline(v, alpha=_default_alpha, ls='--', color='blue')
                else:  # Alfa
                    ax.axhline(v, alpha=_target_alpha, ls='-', color='blue')
    elif means is 'instantaneous':
        # Plot Target and the Average of the other nodes
        _ = plottable[target].plot(ax=ax, alpha=_target_alpha, style='-')
        _ = plottable[[c for c in plottable.columns if c != target]].mean(axis=1).plot(ax=ax, alpha=_default_alpha,
                                                                                       style='--')
    else:
        raise RuntimeError("There should have been no way to get here; this case was supposed to be caught on launch")

    if stds:
        map(pltstd, zip(result.mean(), result.std()))
    ax.set_xlabel("Mission Time (mins)")
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_xticklabels(np.arange(0, 61, 10))
    ax.set_ylabel("Weighted Trust Value")
    ax.set_ylim([0.0, 1.0])
    ax.legend().set_visible(False)

    if box is not None:

        _boxplot_kwargs = dict(
            widths=0.8,
            showmeans=True,
            meanline=True,
            meanprops=dict(linestyle='-', linewidth=0.5, alpha=0.5),
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5, color='black', alpha=0.5),
            return_type='dict'
        )

        axb = plt.subplot(gs[1])
        if box is 'summary':
            bp = axb.boxplot([dropna(plottable.iloc[:, 0].values), plottable.iloc[:, 1:].stack().values],
                        labels=['Misbehaver', 'Other Nodes'], **_boxplot_kwargs)
        elif box is 'complete':
            bp = plottable.boxplot(rot=90, ax=axb, **_boxplot_kwargs)
            for i,c in enumerate(plottable.columns):
                plt.setp(bp['boxes'][i], color=palette[c])
                plt.setp(bp['means'][i], color=palette[c])
                plt.setp(bp['medians'][i], color=palette[c])

        index_width = len(result.columns)
        target_x = (0.5 + target_index) / index_width
        axb.annotate('', xy=(target_x, 0.95), xycoords='axes fraction', xytext=(target_x, 1.05),
                     arrowprops=dict(arrowstyle="->", color='r', linewidth=1))

        plt.setp(axb.get_yticklabels(), visible=False)
        axb.grid(b=False)

    fig.tight_layout()

    if plot_filename is not None:
        savefig(fig, name=plot_filename, ax=[ax,axb], transparent=True)
        plt.close(fig)

    return fig, [ax, axb], result


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
            plot_trust_line_graph(run, title="{0}{1}".format(bev_key, run_i))
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
        this_title = "{0} {1}".format(title, '-'.join(key))
        ax.set_title(this_title)
        format_axes(ax)
        fig.tight_layout()
        yield (fig, ax)


def format_features(feats):
    alt_feats = pd.concat(feats, names=['base', 'comp', 'metric']).unstack('metric')
    alt_feats.index.set_levels(
        [[u'MPC', u'STS', u'Fair', u'Shadow', u'SlowCoach'], [u'MPC', u'STS', u'Fair', u'Shadow', u'SlowCoach']],
        inplace=True)
    return alt_feats


def best_run_and_weight(f, trust_observations, par=True, tolerance=0.01):
    f = pd.Series(f, index=trust_observations.keys())
    f_val = f.values

    @Tools.timeit()
    def generate_weighted_trust_perspectives(_trust_observations, feat_weights, par=True):
        weighted_trust_perspectives = []
        for w in feat_weights:
            weighted_trust_perspectives.append(
                Trust.generate_node_trust_perspective(
                    _trust_observations,
                    metric_weights=pd.Series(w),
                    par=par
                ))
        return weighted_trust_perspectives

    def _assess(x):
        return -np.subtract(*map(np.nanmean, np.split(x.values, [1], axis=1)))

    def best_group_in_perspective(perspective):
        group = perspective.groupby(level=['observer', 'run']) \
            .apply(_assess)
        best_group = group.argmax()
        return best_group, group[best_group]

    combinations = np.asarray([f_val * i for i in itertools.product([-1, 1], repeat=len(f))])
    for i in f.values[np.abs(f_val) < tolerance]:
        combinations[:, np.where(f_val == i)] = i
    combinations = Tools.npuniq(combinations)

    print("Have {0} Combinations".format(len(combinations)))
    perspectives = generate_weighted_trust_perspectives(trust_observations,
                                                        combinations, par=par)
    print("Got Perspectives")
    group_keys, assessments = zip(*map(best_group_in_perspective, perspectives))
    best_weight = combinations[np.argmax(assessments)]
    best_run = group_keys[np.argmax(assessments)]
    best_score = np.max(assessments)
    print("Winner is {0} with {1}@{2}".format(best_run, best_weight, best_score))
    if np.all(best_weight == f):
        print("Actually got it right first time for a change!")
    return best_run, best_weight


def best_of_all(feats, trust_observations, par=True):
    best = defaultdict(dict)
    for (base_str, target_str), feat in feats.to_dict().items():
        if base_str != "Fair":
            continue
        print(base_str)

        print("---" + target_str)
        best[base_str][target_str] = \
            best_run_and_weight(
                feat,
                trust_observations,
                par=par)
    return best


def metric_subset_analysis(trust_observations, key, subset_str, weights_d=None,
                           plot_internal=False, par_ctx=None, node_palette=None,
                           texcol=_texcol):
    # alt_ indicates using a (presumably) non malicious node as the target of trust assessment.
    weights = {}
    trust_perspectives = {}
    time_meaned_plots = {}
    alt_time_meaned_plots = {}
    instantaneous_meaned_plots = {}
    alt_instantaneous_meaned_plots = {}

    if key is None:
        _trust_observations = trust_observations
    else:
        _trust_observations = trust_observations[key]
    # Get the best results for graph generation (from pubscripts.__init__.best_of_all)
    if weights_d is None:
        best_d = Tools.uncpickle(fig_basedir + '/best_{0}_runs'.format(subset_str))['Fair']
    else:
        try:
            best_d = weights_d[subset_str]['Fair']
        except:
            print("No Fair Weight in {}, skipping".format(subset_str))
            return None

    for target_str, best in best_d.items():
        trust_perspective = Trust.generate_node_trust_perspective(
            _trust_observations.xs(target_str, level='var'),
            metric_weights=pd.Series(best[1], dtype=np.float64),
            par=False)

        weights[(subset_str, target_str)] = np.asarray(best[1])

        fig_filename = "best_{0}_run_time_{1}".format(subset_str, target_str) if plot_internal else None
        time_meaned_plots[(subset_str, target_str)] = plot_trust_line_graph(trust_perspective \
                                                                            .xs(best[0],
                                                                                level=['observer', 'run']) \
                                                                            .dropna(axis=1, how='all'),
                                                                            stds=False,
                                                                            spans=6,
                                                                            box='complete',
                                                                            means='time',
                                                                            _texcol=texcol,
                                                                            _texfac=_texfac,
                                                                            plot_filename=fig_filename,
                                                                            palette=node_palette
                                                                            )

        fig_filename = "best_{0}_run_instantaneous_{1}".format(subset_str, target_str) if plot_internal else None
        instantaneous_meaned_plots[(subset_str, target_str)] = plot_trust_line_graph(trust_perspective \
                                                                                     .xs(best[0],
                                                                                         level=['observer',
                                                                                                'run']) \
                                                                                     .dropna(axis=1, how='all'),
                                                                                     stds=False,
                                                                                     spans=6,
                                                                                     box='complete',
                                                                                     means='instantaneous',
                                                                                     _texcol=texcol,
                                                                                     _texfac=_texfac,
                                                                                     plot_filename=fig_filename,
                                                                                     palette=node_palette
                                                                                     )
        # Plotting using an alternate observer for completeness (presumably bravo)
        # alt_target = 'Bravo' if 'Bravo' != target else 'Charlie'
        if 'Bravo' != best[0][0]:
            alt_target = 'Bravo'
        else:
            alt_target = 'Charlie'
        fig_filename = "best_{0}_run_alt_time_{1}".format(subset_str, target_str) if plot_internal else None
        alt_time_meaned_plots[(subset_str, target_str)] = plot_trust_line_graph(trust_perspective \
                                                                                .xs(best[0],
                                                                                    level=['observer', 'run']) \
                                                                                .dropna(axis=1, how='all'),
                                                                                target=alt_target,
                                                                                stds=False,
                                                                                spans=6,
                                                                                box='complete',
                                                                                means='time',
                                                                                _texcol=texcol,
                                                                                _texfac=_texfac,
                                                                                plot_filename=fig_filename,
                                                                                palette=node_palette
                                                                                )

        fig_filename = "best_{0}_run_alt_instantaneous_{1}".format(subset_str, target_str) if plot_internal else None
        alt_instantaneous_meaned_plots[(subset_str, target_str)] = plot_trust_line_graph(trust_perspective \
                                                                                         .xs(best[0],
                                                                                             level=['observer',
                                                                                                    'run']) \
                                                                                         .dropna(axis=1,
                                                                                                 how='all'),
                                                                                         target=alt_target,
                                                                                         stds=False,
                                                                                         spans=6,
                                                                                         box='complete',
                                                                                         means='instantaneous',
                                                                                         _texcol=texcol,
                                                                                         _texfac=_texfac,
                                                                                         plot_filename=fig_filename,
                                                                                         palette=node_palette
                                                                                         )

        plt.close('all')
        trust_perspectives[(subset_str,target_str)] = trust_perspective.copy()

    if plot_internal:
        return dict(weights=weights,
                    trust_perspectives=trust_perspectives
        )
    else:
        return dict(
            trust_perspectives=trust_perspectives,
            time_meaned_plots=time_meaned_plots,
            alt_time_meaned_plots=alt_time_meaned_plots,
            instantaneous_meaned_plots=instantaneous_meaned_plots,
            alt_instantaneous_meaned_plots=alt_instantaneous_meaned_plots,
            weights=weights
        )
