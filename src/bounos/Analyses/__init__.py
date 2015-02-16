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
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import logging
import os

import pandas as pd

import bounos.ChartBuilders as ChartBuilders


FORMAT = "%(asctime)-10s %(message)s"
logging.basicConfig(format=FORMAT,
                    level=logging.INFO,
                    datefmt='%H:%M:%S',
                    filename="/dev/shm/multi_loader.log")

log = logging.getLogger()


def get_valid_metric(metric):
    """
    Returns an instantiated, valid, metric if possible.

    Inspects the TOP of the MRO of an attempted class to check if it matches the class name (i.e. not the id)

    issubclass doesn't work across execution instances  (i.e. saved data files / parralel execution)

    Class structure of metric is expected to be [ something -> bounos.Metrics.Metric -> object ]
    :param metric:
    :return:
    """
    import bounos.Metrics as Metrics

    metric_arg = metric
    if isinstance(metric_arg, type) and hasattr(metric_arg, 'mro') and metric_arg.mro()[-2].__name__ == 'Metric':
        metric_class = metric_arg
    elif isinstance(metric_arg, str):
        metric_class = getattr(Metrics, metric_arg)
    else:
        raise ValueError(
            "Invalid object give for metric, should be either subclass of bounos.Metrics.Metric or string: got type {} containing {}:{}".format(
                type(metric_arg), metric_arg, metric_arg.__bases__
            ))
    metric = metric_class()
    if metric is None:
        raise ValueError("No Metric! Cannot Contine")
    return metric


def hdf_process_kickstart(logstore, directory, keys):
    """
    Helper Function for HDF processing methods providing
        a: It's own key (self awareness is great) (first if multiple)
        b: A dataframe/dir or frames associated with a particular key/keys
        c: a fully set up path to somewhere to go depending on what directory it's been given

    :param logstore:
    :param directory:
    :return: keystring, df, path
    """

    if isinstance(keys, list):
        keystring = keys[0]
    else:
        keystring = keys

    if directory:
        # base path already exists so subdir
        path = os.path.join(directory, keystring)
        os.makedirs(path, exist_ok=True)
    else:
        # Spam all the directories
        path = keystring
        os.makedirs(path, exist_ok=True)

    with pd.get_store as store:
        if isinstance(keys, list):
            df = {k: store.get(k) for k in keys}
        else:
            df = store.get(keys)

    return keystring, df, path


def process_all_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/rx', '/stats', '/trust', '/trust_accessories', '/tx', '/tx_queue']
    :param logstore: str
    :param title: str
    :param directory: assumes pwd if not given AND valid
    :return:
    """
    processes = [
        process_stats_logstore_graphics,
        process_rx_logstore_graphics,
        process_tx_logstore_graphics,
        process_tx_queue_logstore_graphics,
        process_trust_logstore_graphics
    ]

    for process in processes:
        try:
            process(logstore, title, directory)
        except:
            log.exception("Failed on {}".format(process.__name__))


def process_stats_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/stats']
    :param logstore: str
    :param title: str
    :param directory: assumes pwd if not given AND valid
    :return:
    """
    keystring, df, path = hdf_process_kickstart(logstore, directory, 'stats')

    ChartBuilders.performance_summary_for_variable_packet_rates(df)


def process_rx_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/rx']
    :param logstore: str
    :param title: str
    :param directory: assumes pwd if not given AND valid
    :return:
    """
    keystring, df, path = hdf_process_kickstart(logstore, directory, 'stats')


def process_tx_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/tx']
    :param logstore: str
    :param title: str
    :param directory: assumes pwd if not given AND valid
    :return:
    """
    keystring, df, path = hdf_process_kickstart(logstore, directory, 'stats')
    ChartBuilders.lost_packets_by_sender_reciever(df)


def process_tx_queue_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/tx_queue']
    :param logstore: str
    :param title: str
    :param directory: assumes pwd if not given AND valid
    :return:
    """
    keystring, df, path = hdf_process_kickstart(logstore, directory, 'stats')


def process_trust_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/trust', '/trust_accessories']
    :param logstore: str
    :param title: str
    :param directory: assumes pwd if not given AND valid
    :return:
    """
    keystring, dfs, path = hdf_process_kickstart(logstore, directory, ['trust', 'trust_associates'])
