#!/usr/bin/env python
# coding=utf-8
"""
 * This file is part of the Aietes Framework (https://github.com/andrewbolster/aietes)
 *
 * (C) Copyright 2013 Andrew Bolster (http://andrewbolster.info/) and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import bounos

_boxplot_kwargs = {
    # 'showmeans': False,
    # 'showbox': True,
    # 'widths': 0.6,
    # 'linewidth': 2,
    'showfliers': False,
    "whis": 1
}


def lost_packet_distribution(dp=None, tx=None, title=None):
    """
    Return a dist+rug plot of lost packets for the given DataPackage
    :param dp: bounos.DataPackage
    :return f: plt.figure
    """
    # Fancy XOR - http://stackoverflow.com/questions/432842/how-do-you-get-the-logical-xor-of-two-variables-in-python
    if (dp is None) == (tx is None):
        raise ValueError("Need either dp={} or rx={}, not both".format(
            type(bounos.DataPackage),
            type(pd.DataFrame)
        ))
    elif tx is None:
        assert isinstance(dp, bounos.DataPackage)
        df = dp.get_global_packet_logs(pkt_type='tx')
        title = dp.title
        tmax = dp.tmax
    elif dp is None:
        assert isinstance(tx, pd.DataFrame)
        df = tx
        title = "FIXME" if title is None else title
        tmax = int(np.ceil(
            tx.time_stamp.max() / 3600) * 3600)  # assume we're not masochists and it's rounded up to an hour within
    else:
        raise ValueError("I've got no idea how you got here...")

    died = df[not df.delivered].count().max()
    if died > 1:
        all_pkts = df.count().max()

        f, ax = plt.subplots(figsize=(13, 7))
        ax.set_title(
            "Distribution of lost packets over time for {} model: total={:.2%} of {}".format(title,
                                                                                             died / float(all_pkts),
                                                                                             all_pkts))
        ax.set_ylabel("Count (n)")
        ax.set_xlabel("Simulation Time (s)")
        with np.errstate(invalid='ignore'):
            sns.distplot(df.time_stamp[df.delivered != True], kde=False, rug=True, bins=10, ax=ax)
    else:
        raise ValueError("Not enough lost packets")

    return f


def end_to_end_delay_distribution(dp):
    """
    Return a per-node overlaid end to end delay distribution chart
    :param dp: bounos.DataPackage
    :return f: plt.figure
    """
    assert isinstance(dp, bounos.DataPackage)
    df = dp.get_global_packet_logs(pkt_type='rx')

    f, ax = plt.subplots(figsize=(13, 7))
    nsources = len(df.source.unique())
    c = sns.color_palette("Set1", nsources)
    ax.set_title("End-To-End Delay Received Distribution for {} model: average={:.2f}s".format(dp.title, float(
        df[["delay"]].mean())))

    for i, source in enumerate(sorted(df.source.unique())):
        sns.distplot(df[df.source == source]['delay'], color=c[i], ax=ax, label=source, kde=False)
    ax.set_ylabel("Count (n)")
    ax.set_xlabel("Delay (s)")
    plt.legend()
    return f


def source_and_dest_delay_violin_plot(dp):
    """
    Return a plot of the source and destination RX delay violin plots as two subplots.
    :param dp:
    :return:
    """
    assert isinstance(dp, bounos.DataPackage)
    df = dp.get_global_packet_logs(pkt_type='rx')
    f, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 4))
    sns.violinplot(df.delay, df.source, color="Paired", ax=ax_l)
    sns.violinplot(df.delay, df.dest, color="Paired", ax=ax_r)

    f.suptitle("Per-Node End-to-End Delay Received Distribution for {} model: average={:.2f}s".format(dp.title, float(
        df[["delay"]].mean())))

    ax_l.set_xlabel("Source Node")
    ax_r.set_xlabel("Destination Node")
    ax_r.set_ylabel("Delay (s)")
    ax_l.set_ylabel("Delay (s)")
    return f


def source_and_dest_delay_cdf_plot(dp=None, rx=None, title=None, figsize=(12, 4)):
    """
    Return a plot of the source and destination RX delay CDF as two subplots.
    :param dp:
    :return:
    """
    # Fancy XOR - http://stackoverflow.com/questions/432842/how-do-you-get-the-logical-xor-of-two-variables-in-python
    if (dp is None) == (rx is None):
        raise ValueError("Need either dp={} or rx={}, not both".format(
            type(bounos.DataPackage),
            type(pd.DataFrame)
        ))
    elif rx is None:
        assert isinstance(dp, bounos.DataPackage)
        df = dp.get_global_packet_logs(pkt_type='rx')
        title = dp.title
        tmax = dp.tmax
    elif dp is None:
        df = rx
        title = "FIXME" if title is None else title
        tmax = int(np.ceil(
            rx.received.max() / 3600) * 3600)  # assume we're not masochists and it's rounded up to an hour within
    else:
        raise ValueError("I've got no idea how you got here...")

    f, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize)
    f.suptitle("Per-Node End-to-End Delay Cumulative Distribution for {} model: average={:.2f}s".format(title, float(
        df[["delay"]].mean())))

    # Delays groups by source
    for i, group in df.delay.groupby(df.source):
        values, base = np.histogram(group, bins=40)
        # evaluate the cumulative
        cumulative = np.cumsum(values)
        # plot the cumulative function
        ax_l.plot(base[:-1], cumulative, label=i)
    ax_l.set_title("Source Node")
    ax_l.set_xlabel("Delay (s)")
    ax_l.legend(loc="lower right")
    ax_l.set_ylabel('Packet Count')

    # Delays groups by destination
    for i, group in df.delay.groupby(df.dest):
        values, base = np.histogram(group, bins=40)
        # evaluate the cumulative
        cumulative = np.cumsum(values)
        # plot the cumulative function
        ax_r.plot(base[:-1], cumulative, label=i)
    ax_r.set_title("Destination Node")
    ax_r.set_xlabel("Delay (s)")
    ax_r.legend(loc="lower right")

    return f


def _channel_occupancy_calc(df):
    tx = pd.DataFrame(index=df.time_stamp)
    rx = pd.DataFrame(index=df.received)
    tx['p'] = 1  # adding a packet
    rx['p'] = -1  # receiving a packet

    # create the time series here
    t = pd.concat([tx, rx])
    t.index = pd.to_datetime(t.index, unit='s')  # to convert from nanoseconds to seconds
    return t.resample('s', how='sum').cumsum().fillna(0)


def channel_occupancy_distribution(dp=None, rx=None, title=None, figsize=(13, 7)):
    """
    Generates a plot showing the distribution of the number of in-the-air packets at each time step

    :param rx:
    :param title:
    :param figsize:
    http://stackoverflow.com/questions/26760962/python-pandas-count-of-records-between-values-at-a-given-time-packets-in-fli

    This *should* be run per-dataset or per-run but in the RX case is happy enough to take a var/run/node/n multiindex,
     the results just make no sense...

    :param dp:
    :return: f plt.figure
    """

    # Fancy XOR - http://stackoverflow.com/questions/432842/how-do-you-get-the-logical-xor-of-two-variables-in-python
    if (dp is None) == (rx is None):
        raise ValueError("Need either dp={} or rx={}, not both".format(
            type(bounos.DataPackage),
            type(pd.DataFrame)
        ))
    elif rx is None:
        assert isinstance(dp, bounos.DataPackage)
        df = dp.get_global_packet_logs(pkt_type='rx')
        title = dp.title
        tmax = dp.tmax
    elif dp is None:
        df = rx
        title = "FIXME" if title is None else title
        tmax = int(np.ceil(
            rx.received.max() / 3600) * 3600)  # assume we're not masochists and it's rounded up to an hour within
    else:
        raise ValueError("I've got no idea how you got here...")

    if 'var' in df.index.names and len(df.groupby(level='var')) > 1:
        raise ValueError(
            "It's really not a good idea for multiple var's to go into this; groupby-apply this instead {}".format(
                len(df.index.levels[df.index.names.index('var')])
            ))

    if 'run' in df.index.names:
        channel_occupancy = df.groupby(level=['var', 'run']).apply(_channel_occupancy_calc).unstack('run').sum(axis=1)
    else:
        channel_occupancy = _channel_occupancy_calc(df).p  # should work fine for a single run
    f, ax = plt.subplots(figsize=figsize)
    ax.set_title("In-The-Air packets for {} model: Channel Use={:.2%}".format(
        title, sum(channel_occupancy > 0) / float(channel_occupancy.size)))
    ax.set_ylabel("Percentage of Runtime (%)")
    ax.set_xlabel("Number of packets in the medium (n)")
    channel_occupancy.hist()
    formatter = lambda f, p: "{:.0%}".format(f / float(tmax))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))

    return f


def combined_trust_observation_summary(dp=None, trust_log=None, pos_log=None, target=None, title=None,
                                       sampleperiod=None, metric_weights=None):
    """

    :param dp:
    :param trust_log:
    :param pos_log:
    :param target:
    :param title:
    :param sampleperiod:
    :param metric_weights:
    :return: :raise ValueError:
    """
    from bounos.Analyses import Trust

    # TODO THIS HAS NOT BEEN FIXED FOR DATAFRAME MGMT

    # Fancy XOR - http://stackoverflow.com/questions/432842/how-do-you-get-the-logical-xor-of-two-variables-in-python
    if (dp is None) == (trust_log is None or pos_log is None):
        raise ValueError("Need either dp={} or trust_log={} and pos_log={}, not both".format(
            type(bounos.DataPackage),
            type(pd.DataFrame),
            type(pd.DataFrame)
        ))
    elif trust_log is None:
        assert isinstance(dp, bounos.DataPackage)
        trust_log = dp.get_global_trust_logs()
        trust = Trust.generate_global_trust_values(trust_log, metric_weights=metric_weights)
        title = dp.title
        tmax = dp.tmax
        names = dp.names
    elif dp is None:
        assert isinstance(trust_log, pd.DataFrame)
        assert isinstance(pos_log, pd.DataFrame)
        title = "FIXME" if title is None else title
        tmax = int(np.ceil(
            trust_log.received.max() / 3600) * 3600)  # assume we're not masochists and it's rounded up to an hour within
    else:
        raise ValueError("I've got no idea how you got here...")

    if target is None:
        target = 'n1'

    f, ax = plt.subplots(len(names), 3, sharex=False, figsize=(16, 16))
    lines = []

    f.suptitle("Trust overview for target {} (highlighted) in {} model".format(target, dp.title))
    for i, i_node in enumerate(names):
        map(lambda (k, v): lines.append(ax[i][0].plot(v, label=k, alpha=0.5 if k != target else 0.9)),
            trust[i_node].items())
        ax[i][0].legend(loc="lower center", mode="expand", ncol=6)
        ax[i][0].set_title("Trust Perspective: {}".format(i_node))
        ax[i][0].set_ylim(0, 1)

        ax[i][1].set_title("Distance to {}: {}".format(target, i_node))
        ax[i][1].plot(np.linalg.norm(dp.p[i, :, ::600] - dp.p[dp.names.index(target), :, ::600], axis=0))
        ax[i][1].set_ylim(0, 300)

        ax[i][2].set_title("Distribution of raw trust values from {}".format(i_node))
        with np.errstate(invalid='ignore'):
            sns.boxplot(
                pd.DataFrame.from_dict({key: pd.Series(vals) for key, vals in trust[i_node].items()}),
                ax=ax[i][2], showmeans=True, showbox=False, widths=0.2, linewidth=2)
            ax[i][2].legend()
            ax[i][2].set_ylim(0, 1.0)

        # Harrang Labels
        if i + 1 < dp.n:
            plt.setp(ax[i][0].get_xticklabels(), visible=False)
            plt.setp(ax[i][1].get_xticklabels(), visible=False)
            plt.setp(ax[i][2].get_xticklabels(), visible=False)
        else:
            ax[i][0].set_xlabel("Trust Observation Iterations (600s)")
            ax[i][1].set_xlabel("Trust Observation Iterations (600s)")
            ax[i][2].set_xlabel("Trust Opinion of node")

    return f


def performance_summary_for_var(stats, title=None, var='Packet Rates', rename_labels=None, figsize=None,
                                hide_annotations=False):
    """

    :param stats:
    :param title:
    :param var:
    :param figsize:
    :return:
    """
    labels = {s: s for s in stats.keys()}
    labels.update(rename_labels)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    grp = stats.groupby(level='var')[['rx_counts', 'enqueued', 'collisions']].mean()
    if rename_labels:
        grp.rename(columns=rename_labels, inplace=True)

    grp.index = grp.index.astype(np.float64)
    grp.plot(ax=ax,
             secondary_y=labels['collisions'],  # subplots=True,
             style=["-", "--", "-.", ":"],
             grid='on',
             title="Performance Comparison of Varying {},{}".format(var, (':' + title if title is not None else ""))
             )
    ax.set_xlabel(var)
    ax.set_ylabel("Total Packets")
    if not hide_annotations:
        maxes = stats.groupby(level='var')['rx_counts'].max()
        maxes.index = maxes.index.astype(np.float64)
        differential = maxes.diff()
        diffmax = differential.argmax()
        maxmax = maxes.argmax()
        ax.axvline(diffmax, alpha=0.2, linestyle=':')
        ax.text(diffmax, maxes.mean() / 2, 'd(RX) Heel @ {:.4f}'.format(diffmax), rotation=90)
        ax.axvline(maxmax, alpha=0.2, linestyle=':')
        ax.text(maxmax, maxes.mean() / 2, 'RX Max @ {:.4f}'.format(maxmax), rotation=90)
    return f


def probability_of_timely_arrival(stats, title=None, var='Packet Rates', figsize=None):
    """

    :param stats:
    :param title:
    :param var:
    :param figsize:
    :return:
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    packet_error_rates = (stats.rx_counts / stats.tx_counts).groupby(level='var').mean()
    packet_error_vars = (stats.rx_counts / stats.tx_counts).groupby(level='var').std()
    ax.errorbar(list(packet_error_rates.index), packet_error_rates.values, packet_error_vars)
    ax.set_ylabel("Probability of Timely Arrival")
    ax.set_xlabel(var)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Probability of Timely Arrival of Varying {},{}".format(var, (':' + title if title is not None else "")))
    return f


def average_delays_across_variation(stats, title=None, var='Packet Rates', figsize=None, ylog=False):
    """

    :param stats:
    :param title:
    :param var:
    :param figsize:
    :return:
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    packet_delays = stats.groupby(level='var').average_rx_delay.mean()
    packet_delays_std = stats.groupby(level='var').average_rx_delay.std()
    ax.errorbar(list(packet_delays.index), packet_delays.values, yerr=packet_delays_std.values)
    ax.set_ylabel("Average End to End Delay(s)")
    ax.set_xlabel(var)
    if ylog:
        ax.set_yscale('log')

    # Cannot have negative delay
    ax.set_ylim((0, ax.get_ylim()[1]))

    ax.set_title(
        "Average End to End Delay for Varying {},{}. \n showing standard deviation of result, with a max of {}".format(
            var, (':' + title if title is not None else ""), np.around(np.max(packet_delays_std), decimals=2)))
    return f


def rts_ratio_across_variation(stats, title=None, var='Packet Rates', figsize=None, ylog=False):
    """

    :param stats:
    :param title:
    :param var:
    :param figsize:
    :return:
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    ratio = (stats.rts_counts / stats.rx_counts)
    r_mean = ratio.groupby(level='var').mean()
    r_std = ratio.groupby(level='var').std()
    ax.errorbar(list(r_mean.index), r_mean.values, yerr=r_std.values)
    ax.set_ylabel("RTS/Data Ratio")
    ax.set_xlabel(var)
    if ylog:
        ax.set_yscale('log')

    # Cannot have negative ratio
    ax.set_ylim((0, ax.get_ylim()[1]))

    ax.set_title("RTS / Data Ratio of Varying {},{}. "
                 "\n showing standard deviation of result, with a max of {}".format(
                     var,
                     (':' + title if title is not None else ""),
                     np.around(np.max(r_std.values), decimals=2))
                 )
    return f


def lost_packets_by_sender_reciever(tx, figsize=(16, 13)):
    """
    Can be applied either per var, run, or dataset.
    If Delivered is Nan, the packet is stuck in a tx_queue and they can look after that
    This is concerned with routing level losses
    :param figsize:
    :param tx:
    :return:
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    total_not_queued = tx.delivered.dropna().shape[0]
    failed_senders = tx.query('delivered == False').groupby(['source']).size()
    failed_recievers = tx.query('delivered == False').groupby(['dest']).size()

    ind = np.arange(len(failed_senders.keys()))  # the x locations for the groups
    width = 0.35  # the width of the bars

    rects1 = ax.bar(ind, failed_senders, width, color='r')
    rects2 = ax.bar(ind + width, failed_recievers, width, color='y')

    ax.axhline(failed_senders.mean(), color='b', alpha=0.5)
    # http://matplotlib.1069221.n5.nabble.com/Label-for-axhline-td39870.html
    tick = ax.set_yticks([failed_senders.mean()], minor=True)[0]
    tick.label1On = False
    tick.label2On = True
    ax.set_yticklabels([int(failed_senders.mean())], minor=True, verticalalignment='center')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Packets Not Delivered (% of total transmitted)')
    ax.set_title('Lost Packets by Sender and Recepient')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((failed_senders.keys()))

    formatter = lambda f, p: "{:.4%}".format(f / float(total_not_queued))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))

    ax.legend((rects1[0], rects2[0]), ('Source', 'Destination'))

    def autolabel(rects):
        # attach some text labels
        """

        :param rects:
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    return f


def trust_perspectives_wrt_observers(trust_frame, title=None, figsize=(16, 2)):
    """
    Generates a 'matrix' of trust assessments from each nodes perspective to every other one, grouped by 'var'
    :param title:
    :param figsize:
    :param trust_frame:
    :return:
    """
    groups = trust_frame.groupby(level=['var'])
    n_nodes = trust_frame.shape[1]

    f, ax = plt.subplots(len(groups), n_nodes, figsize=(figsize[0], figsize[1] * len(groups)), sharey=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.05, top=0.951)
    for i, (var, group) in enumerate(groups):
        for j, (jvar, jgroup) in enumerate(group.groupby(level='observer')):
            sns.boxplot(jgroup, ax=ax[i][j], **_boxplot_kwargs)
            if not i:  # first plot
                ax[i][j].set_title(jvar)
        map(lambda a: a.set_xlabel(""), ax[i])
        if i + 1 < len(groups):
            ax[i][0].set_xlabel("Target")
        ax[i][0].set_ylabel("{:.4f}".format(float(var)))
    f.suptitle(
        "Plots of Per-Node Subjective Trust Values {}\n(Each sub plot is a single nodes trust viewpoint of other nodes)".format(
            title if title is None else ": " + title
        ), fontsize=24)
    return f


def trust_perspectives_wrt_targets(trust_frame):
    """
    Generates a 'matrix' of trust assessments of each nodes perspective from every other one, grouped by 'var'
    :param trust_frame:
    :return:
    """
    groups = trust_frame.unstack('observer').stack('target').groupby(level=['var'])
    n_nodes = trust_frame.shape[1]

    f, ax = plt.subplots(len(groups), n_nodes, figsize=(16, 2 * len(groups)), sharey=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.05, top=0.951)
    for i, (var, group) in enumerate(groups):
        for j, (jvar, jgroup) in enumerate(group.groupby(level='target')):
            sns.boxplot(jgroup, ax=ax[i][j], **_boxplot_kwargs)
            if not i:  # first plot
                ax[i][j].set_title(jvar)
        map(lambda a: a.set_xlabel(""), ax[i])
        if i + 1 < len(groups):
            ax[i][0].set_xlabel("Observer")
        ax[i][0].set_ylabel("{:.4f}".format(float(var)))
    f.suptitle(
        "Plots of Per-Node Objective Trust Values\n(Each sub plot is a nodes trust value from the perspective of other nodes)",
        fontsize=24)
    return f


def trust_network_wrt_observers(trust_group, var, title=False, figsize=(16, 2), texify=True, xlabel=True,
                                dropnet=False):
    """
    Generates a 'matrix' of trust assessments from each nodes perspective to every other one, grouped by 'var'
    :param title:
    :param figsize:
    :param trust_frame:
    :return:
    """
    df = trust_group.dropna()
    if dropnet:
        df = df.drop("$T_{Net}$", 1)
    f, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 1), sharey=True)
    # plt.subplots_adjust(hspace=0.2, wspace=0.05, top=0.951)
    ax = sns.boxplot(df, ax=ax, **_boxplot_kwargs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.axvline(5.5, ls="--", color="k", alpha=0.5)
    [ax.axvline(x, ls=":", color="k", alpha=0.5) for x in [1.5, 3.5]]
    ax.annotate("D", xy=(1, 0.9), xycoords='data', horizontalalignment='centre')
    ax.annotate("R", xy=(2.5, 0.9), xycoords='data', horizontalalignment='centre')
    ax.annotate("I", xy=(4.5, 0.9), xycoords='data', horizontalalignment='centre')
    if dropnet:
        ax.annotate("Agg.", xy=(6, 0.9), xycoords='data', horizontalalignment='centre')
    else:
        ax.annotate("Agg.", xy=(6.5, 0.9), xycoords='data', horizontalalignment='centre')

    if xlabel:
        ax.set_xlabel("Assessment Type")
    ax.set_ylabel("Trust Value")
    ax.set_ylim((0.0, 1.0))

    f.tight_layout(pad=0.1)
    if title is not False:
        ax.set_title("{}".format(
            var if title is None else ": " + title
        ))
    return f


def plot_axes_views_from_packet_frames(df, title=None, figsize=None):
    """

    :param df:
    :param title:
    :param figsize:
    :return:
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    f.suptitle("Node Layout and mobility {}".format("for " + title if title is not None else ""))
    for n, (name, node_p) in enumerate(df.groupby('source')):
        node_p = pd.DataFrame.from_records(node_p.source_position.values)
        x, y, z = initial = node_p.iloc[0]

        ax1.annotate(name, xy=(x, y), xytext=(-20, 5), textcoords='offset points', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                     color='red')
                     )
        ax1.scatter(x, y)
        ax2.scatter(y, z)
        ax3.scatter(x, z)

        xs = node_p[0]
        ys = node_p[1]
        zs = node_p[2]
        ax1.plot(xs, ys, alpha=0.6)
        ax2.plot(ys, zs, alpha=0.6)
        ax3.plot(xs, zs, alpha=0.6)
        ax4.plot(
            np.abs(node_p.apply(np.linalg.norm, axis=1) - np.linalg.norm(initial)),
            label=name,
            alpha=0.6
        )

    ax1.set_title("X-Y (Top)")
    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_title("Y-Z (Side)")
    ax2.set_aspect('equal', adjustable='datalim')
    ax3.set_title("X-Z (Front)")
    ax3.set_aspect('equal', adjustable='datalim')
    ax4.set_title("Distance from initial position (m)")
    ax4.legend(loc='upper center', ncol=3)
    return f


def plot_axes_views_from_positions_frame(df, title=None, figsize=None):
    """

    :param df:
    :param title:
    :param figsize:
    :return:
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    f.suptitle("Node Layout and mobility {}".format("for " + title if title is not None else ""))
    for n, (name, node_p) in enumerate(df.groupby(level='node')):
        x, y, z = initial = node_p.iloc[0]

        ax1.annotate(name, xy=(x, y), xytext=(-20, 5), textcoords='offset points', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                     color='red')
                     )
        ax1.scatter(x, y)
        ax2.scatter(y, z)
        ax3.scatter(x, z)

        xs = node_p.x
        ys = node_p.y
        zs = node_p.z
        ax1.plot(xs, ys, alpha=0.6)
        ax2.plot(ys, zs, alpha=0.6)
        ax3.plot(xs, zs, alpha=0.6)
        ax4.plot(
            np.abs(node_p.apply(np.linalg.norm, axis=1) - np.linalg.norm(initial)),
            label=name,
            alpha=0.6
        )

    ax1.set_title("X-Y (Top)")
    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_title("Y-Z (Side)")
    ax2.set_aspect('equal', adjustable='datalim')
    ax3.set_title("X-Z (Front)")
    ax3.set_aspect('equal', adjustable='datalim')
    ax4.set_title("Distance from initial position (m)")
    ax4.legend(loc='upper center', ncol=3)
    return f


def plot_positions(d, bounds=None):
    """
    d as a dict of x,y,z positions (list or ndarray)
    bounds as x,y,z extent of the environment
    :param d:
    :param bounds:
    :return:
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    my_d = {}
    for name, pos in d.items():
        x, y, z = initial = pos

        ax1.annotate(name, xy=(x, y), xytext=(-10, 5), textcoords='offset points', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle='->',
                                     color='red')
                     )
        ax1.scatter(x, y)
        ax2.scatter(y, z)
        ax3.scatter(x, z)
        my_d[name] = initial

    if bounds is not None:
        from matplotlib.patches import Rectangle

        x, y, z = bounds
        ax1.add_patch(Rectangle((0, 0), x, y, facecolor="grey", alpha=0.2))
        ax2.add_patch(Rectangle((0, 0), y, z, facecolor="grey", alpha=0.2))
        ax3.add_patch(Rectangle((0, 0), x, y, facecolor="grey", alpha=0.2))

    ax1.set_ylim((-50, y + 50))
    ax2.set_ylim((-50, z + 50))
    ax2.invert_yaxis()
    ax3.set_ylim((-50, y + 50))
    ax3.invert_yaxis()

    ax1.set_xlim((-50, x + 50))
    ax2.set_xlim((-50, y + 50))
    ax3.set_xlim((-50, x + 50))

    ax4.set_visible(False)

    area = area_of_centroid(my_d)
    avg_dist = avg_range(my_d)
    f.suptitle('All units in (m), Average Range:{:.2e}, Area:{:.2e}'.format(np.around(avg_dist), np.around(area)),
               fontsize=18)
    ax1.set_title("X-Y (Top)")
    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_title("Y-Z (Side)")
    ax2.set_aspect('equal', adjustable='datalim')
    ax3.set_title("X-Z (Front)")
    ax3.set_aspect('equal', adjustable='datalim')

    return f


#####
# LATEXIFY from http://nipunbatra.github.io/2014/08/latexify/
# Extended based on http://damon-is-a-geek.com/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib.html
#####

def latexify(columns=1, factor=0.45):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, pts
    fig_height : float,  optional, pts
    columns : {0.5, 1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    max_height_inches = 8.0

    fig_width_in = 3.339 * factor / columns
    fig_height_in = fig_width_in * golden_mean

    if fig_height_in > max_height_inches:
        print("WARNING: fig_height too large:" + fig_height_in +
              "so will reduce to" + max_height_inches + "inches.")
        fig_height_in = max_height_inches

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'font.size': 10,  # was 10/8
              'legend.fontsize': 8,  # was 10,
              'legend.labelspacing': 0.2,
              'legend.borderpad': 0,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width_in, fig_height_in],
              'font.family': 'serif'
              }

    mpl.rcParams.update(params)
    return fig_width_in, fig_height_in


def format_axes(ax, spine_color='gray'):
    ax.grid(True, color=spine_color, alpha=0.2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(spine_color)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=spine_color)

    return ax


def plot_nodes(node_positions, node_links=None, radius=1.0, scalefree=False, square=True, figsize=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    nodes = []
    node_positions = OrderedDict(sorted(node_positions.items()))
    for node, position in node_positions.items():
        x, y, z = position
        ax.scatter(x, y)
        ax.annotate(node, xy=(x, y), xytext=(-10, 5), textcoords='offset points', ha='center', va='bottom')
        ax.add_patch(plt.Circle((x, y), radius=radius, edgecolor='b', fill=False, alpha=0.2))

    if node_links:
        for node, links in node_links.items():
            for link in links:
                x, y = zip(node_positions.values()[node][0:2], node_positions.values()[link][0:2])
                ax.plot(x, y, color='k', lw=1, alpha=0.4, linestyle='--')

    if square:
        ax.set_aspect('equal', adjustable='datalim')
    if scalefree:
        ax = format_axes(ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)

    return fig


def area_of_centroid(d):
    """

    :param d:
    :return:
    """
    df = pd.DataFrame.from_dict({k: v for k, v in d.items()}, orient='index')
    extent = abs(df.min() - df.max())
    if extent[0] < 0:
        return np.prod(extent.values)
    else:
        return np.prod(extent.values[0:2])


def avg_range(d):
    """

    :param d:
    :return:
    """
    return np.average(squareform(pdist(np.asarray(d.values()))))
