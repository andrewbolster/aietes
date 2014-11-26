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
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import logging
import pandas as pd

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
        raise ValueError("Invalid object give for metric, should be either subclass of bounos.Metrics.Metric or string: got type {} containing {}:{}".format(
            type(metric_arg), metric_arg, metric_arg.__bases__
        ))
    metric = metric_class()
    if metric is None:
        raise ValueError("No Metric! Cannot Contine")
    return metric

def process_all_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/rx', '/stats', '/trust', '/trust_accessories', '/tx', '/tx_queue']
    :param logstore: str
    :param title: str
    :param directory:
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

def process_stats_logstore_graphics(logstore,title,directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/stats']
    :param logstore: str
    :param title: str
    :param directory:
    :return:
    """
    with pd.get_store(logstore) as store:
        stats = store.get('stats')

def process_rx_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/rx']
    :param logstore: str
    :param title: str
    :param directory:
    :return:
    """
    with pd.get_store(logstore) as store:
        rx = store.get('rx')

def process_tx_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/tx']
    :param logstore: str
    :param title: str
    :param directory:
    :return:
    """
    with pd.get_store(logstore) as store:
        tx = store.get('tx')

def process_tx_queue_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/tx_queue']
    :param logstore: str
    :param title: str
    :param directory:
    :return:
    """
    with pd.get_store(logstore) as store:
        tx_queue = store.get('tx_queue')

def process_trust_logstore_graphics(logstore, title, directory=None):
    """
    Coordinate large-run processing for memory efficiency (i.e. all trust at once, all tx/rx at once, etc
    Targeted at the HDFstore log storage containing
    ['/trust', '/trust_accessories']
    :param logstore: str
    :param title: str
    :param directory:
    :return:
    """
    with pd.get_store(logstore) as store:
        trust = store.get('trust')
        trust_accessories = store.get('trust_accessories')