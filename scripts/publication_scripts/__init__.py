# coding=utf-8
import itertools
import os
from collections import OrderedDict

import functools

import numpy as np
import pandas as pd
from matplotlib import pylab as plt, cm, cm, cm, cm
from scipy import interpolate as interpolate

from aietes import Tools
from bounos.ChartBuilders import latexify, plot_nodes

in_results = functools.partial(os.path.join, Tools._results_dir)
app_rate_from_path = lambda s: float(".".join(s.split('-')[2].split('.')[0:-1]))
scenario_map = dict(zip(
    [u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static'],
    ['All Mobile', '$n_1$ Static', '$n_1$ Mobile', 'All Static']
))
scenario_order = list(reversed([u'bella_all_mobile', u'bella_allbut1_mobile', u'bella_single_mobile', u'bella_static']))

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6
latexify(columns=2, factor=0.55)

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


def plot_contour_2d(xi, yi, zi, x=None, y=None, var=None, norm=False):
    if x is None:
        x = []
    if y is None:
        y = []

    fig = plt.figure(figsize=(w, golden_mean * w), facecolor='white')
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
        ax.scatter(x, y, color='k', marker='.')

    # ax.clabel(cset,inline=1)
    return fig


def plot_all_funky_stuff(sdf, mobility):
    xt, yt, zt, x, y = interpolate_rate_sep(sdf.dropna(), "throughput")
    fig = plot_contour_2d(xt, yt, zt, x, y, "Per Node Avg. Throughput (bps)")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/throughput_2d_{}.pdf".format(mobility))

    xd, yd, zd, x, y = interpolate_rate_sep(sdf.dropna(), "average_rx_delay")
    fig = plot_contour_2d(xd, yd, zd, x, y, "Average Delay (s)")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/delay_2d_{}.pdf".format(mobility))

    xd, yd, zd, x, y = interpolate_rate_sep(sdf, "tdivdel")
    fig = plot_contour_2d(xd, yd, zd, x, y, "Throughput Delay Ratio")
    fig.tight_layout(pad=0.1)
    fig.savefig("img/2d_ratio_{}.pdf".format(mobility))

    fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
    fig.tight_layout(pad=0.1)
    fig.savefig("img/3d_ratio_{}.pdf".format(mobility), transparent=True, facecolor='white')

    xd, yd, zd, x, y = interpolate_rate_sep(sdf, "co_norm")
    fig = plot_contour_2d(xd, yd, zd, x, y, "Normalised Throughput Delay Product", norm=True)
    fig.tight_layout(pad=0.1)
    fig.savefig("img/2d_normed_product_{}.pdf".format(mobility))

    fig = plot_contour_3d(xd, yd, zd, rot=45, labels={'x': 'pps', 'y': 'm', 'z': ''})
    fig.tight_layout(pad=0.1)
    fig.savefig("img/3d_normed_product_{}.pdf".format(mobility), transparent=True, facecolor='white')

    fig = plot_lines_of_throughput(sdf)
    fig.tight_layout(pad=0.1)
    fig.savefig("img/throughput_sep_lines_{}.pdf".format(mobility), transparent=True, facecolor='white')


def get_mobility_stats(mobility):
    statsd = OrderedDict()
    trustd = OrderedDict()
    norm = lambda df: (df - np.nanmin(df)) / (np.nanmax(df) - np.nanmin(df))
    rate_and_ranges = filter(
        lambda p: os.path.basename(p).startswith("CommsRateAndRangeTest-bella_{}".format(mobility)),
        result_h5s_by_latest)
    if not rate_and_ranges:
        raise ValueError("No Entries with mobility {}".format(mobility))
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