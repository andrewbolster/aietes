#!/usr/bin/env python
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

__author__ = 'andrewbolster'

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import bounos

def lost_packet_distribution(dp):
    """
    Return a dist+rug plot of lost packets for the given DataPackage
    :param dp: bounos.DataPackage
    :return f: plt.figure
    """
    assert isinstance(dp, bounos.DataPackage)

    df = dp.get_global_packet_logs(pkt_type='tx')

    died = df[df.delivered == False].count().max()
    all_pkts = df.count().max()

    f, ax = plt.subplots(figsize=(13, 7))
    ax.set_title("Distribution of lost packets over time for {} model: total={:.2%} of {}".format(dp.title,died/float(all_pkts), all_pkts))
    ax.set_ylabel("Count (n)")
    ax.set_xlabel("Simulation Time (s)")
    sns.distplot(df.time_stamp[df.delivered == False], kde=False, rug=True, bins=10, ax=ax)
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
    nsources=len(df.source.unique())
    c = sns.color_palette("Set1",nsources)
    ax.set_title("End-To-End Delay Received Distribution for {} model: average={:.2f}s".format(dp.title,float(df[["delay"]].mean())))

    for i,source in enumerate(sorted(df.source.unique())):
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
    f, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True, sharex=True,figsize=(12, 4))
    sns.violinplot(df.delay, df.source, color="Paired", ax=ax_l)
    sns.violinplot(df.delay, df.dest, color="Paired", ax=ax_r)

    f.suptitle("Per-Node End-to-End Delay Received Distribution for {} model: average={:.2f}s".format(dp.title,float(df[["delay"]].mean())))

    ax_l.set_xlabel("Source Node")
    ax_r.set_xlabel("Destination Node")
    ax_r.set_ylabel("Delay (s)")
    ax_l.set_ylabel("Delay (s)")
    return f


def channel_occupancy_distribution(dp):
    """
    Generates a plot showing the distribution of the number of in-the-air packets at each time step

    This is extrememly computationally intensive, taking around 4 seconds for every hour of data (6 nodes)
    and is probably also exponential in complexity
    http://stackoverflow.com/questions/26760962/python-pandas-count-of-records-between-values-at-a-given-time-packets-in-fli

    :param dp:
    :return: f plt.figure
    """
    assert isinstance(dp, bounos.DataPackage)
    df = dp.get_global_packet_logs(pkt_type='rx')

    t_packets = pd.Series([df[(df.time_stamp < t) & (t < df.received)]['source'].count() for t in range(dp.tmax)])
    channeloccupancy = len(np.where(t_packets.values < 1)[0]) / float(dp.tmax)
    f, ax = plt.subplots(figsize=(13, 7))
    ax.set_title("In-The-Air packets for {} model: Channel Use={:.2%}".format(dp.title, 1 - channeloccupancy))
    ax.set_ylabel("Percentage of Runtime (%)")
    ax.set_xlabel("Number of packets in the medium (n)")
    ax.hist(t_packets.values)
    formatter = lambda f, p: "{:.0%}".format(f / float(dp.tmax))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))

    return f



