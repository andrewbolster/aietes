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
n_metrics = 6

from collections import OrderedDict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# THESE LAMBDAS PERFORM GRAY CLASSIFICATION BASED ON GUO
# Still have no idea where sigma comes into it but that's life
_white_fs = [lambda x: -x + 1,
             lambda x: -2 * x + 2 if x > 0.5 else 2 * x,
             lambda x: x
]
_white_sigmas = [0.0, 0.5, 1.0]

_gray_whitenized = lambda x: map(lambda f: f(x), _white_fs)


def _gray_class(x):
    try:
        return _gray_whitenized(x).index(max(_gray_whitenized(x)))
    except ValueError:
        return np.nan


def t_kt(interval):
    """
    Generate a single trust value from a GRG
    1/ (1+ sigma^2/theta^2)
    :param interval:
    :return:
    """
    # good, bad
    theta, sigma = interval
    with np.errstate(divide='ignore'):
        return 1.0 / (
            1.0 + (sigma * sigma) / (theta * theta)
        )


def weight_calculator(metric_index, ignore=None):
    """
    Helper function to take a given Pandas index and return an ordered ndarray vector of balanced (currently)
    weights to apply to the node trust value
    :param metric_index: (i.e. df.keys())
    :param ignore: list of strings of index keys to ignore (i.e. ['blah']
    :return:
    """
    bin_weight = np.asarray(map(lambda k: int(k not in ignore), metric_index))
    return bin_weight / float(np.sum(bin_weight))


def generate_single_run_trust_perspective(gf, metric_weights=None, flip_metrics=None, rho=0.5):
    trusts = []
    for ki, gi in gf.groupby(level='t'):
        gmx = gi.max()
        gmn = gi.min()
        width = gmx - gmn
        with np.errstate(invalid='ignore'):
            good = gi.apply(
                lambda o: (0.75 * np.divide(width, (np.abs(o - gmn)) + rho * width) - 0.5).fillna(1),
                axis=1
            )
            bad = gi.apply(
                lambda o: (0.75 * np.divide(width, (np.abs(o - gmx)) + rho * width) - 0.5).fillna(1),
                axis=1
            )

        if flip_metrics:
            good[flip_metrics], bad[flip_metrics] = bad[flip_metrics], good[flip_metrics]

        interval = pd.DataFrame.from_dict({
            'good': good.apply(np.average, weights=metric_weights, axis=1),
            'bad': bad.apply(np.average, weights=metric_weights, axis=1)
        })
        trusts.append(
            pd.Series(
                interval.apply(
                    t_kt,
                    axis=1),
                name='trust'
            )
        )
    return trusts


def generate_node_trust_perspective(tf, metric_weights=None, flip_metrics=None, rho=0.5, fillna=True, par=True):
    """
    Generate Trust Values based on a big trust_log frame (as acquired from multi_loader or from explode_metrics_...
    Will also accept a selectively filtered trust log for an individual run
    i.e node_trust where node_trust is the inner result of:
        trust.groupby(level=['var','run','node'])


    Parallel Performs significantly better on large(r) datasets, i.e. multi-var.
    ie. Linear- ~2:20s/run vs 50s/run parallel

    :param node_observations: node observations [t][target][x,x,x,x,x,x]
    :param metric_weights: per-metric weighting array (default None)
    :param n_metrics: number of metrics assessed in each observation
    :return:
    """
    assert isinstance(tf, pd.DataFrame)
    trusts = []

    if flip_metrics is None:
        flip_metrics = ['ADelay', 'PLR']

    exec_args = {'metric_weights': metric_weights, 'flip_metrics': flip_metrics, 'rho': rho}

    if par:
        trusts = Parallel(n_jobs=-1)(delayed(generate_single_run_trust_perspective)
                                     (g, **exec_args) for k, g in tf.dropna().groupby(level=['var', 'run', 'observer'])
        )
        trusts = [item for sublist in trusts for item in sublist]
    else:
        for k, g in tf.dropna().groupby(level=['var', 'run', 'observer']):
            trusts.extend(generate_single_run_trust_perspective(g, **exec_args))

    tf = pd.concat(trusts)
    tf.index = pd.MultiIndex.from_tuples(tf.index, names=['var', 'run', 'observer', 't', 'target'])
    tf.index = tf.index.set_levels([
        tf.index.levels[0].astype(np.float64),  # Var
        tf.index.levels[1].astype(np.int32),  # Run
        tf.index.levels[2],  #Observer
        tf.index.levels[3].astype(np.int32),  # T (should really be a time)
        tf.index.levels[4]  #Target
    ])
    tf.sort(inplace=True)

    # The following:
    # Transforms the target id into the column space,
    # Groups each nodes independent observations together
    #   Fills in the gaps IN EACH ASSESSMENT with the previous assessment of that node by that node at the previous time
    if fillna:
        tf = tf.unstack('target').groupby(level=['var', 'run', 'observer']).apply(lambda x: x.fillna(method='ffill'))
    else:
        tf = tf.unstack('target')

    return tf


def invert_node_trust_perspective(node_trust_perspective):
    """
    Invert Node Trust Records to unify against time, i.e. [observer][t][target]
    :param node_trust_perspective:
    :return:
    """
    # trust[observer][t][target] = T_jkt
    trust_inverted = {}
    for j_node in node_trust_perspective[-1].keys():
        trust_inverted[j_node] = np.array([0.5 for _ in range(len(node_trust_perspective))])
        for t in range(len(node_trust_perspective)):
            if t < len(node_trust_perspective) and node_trust_perspective[t].has_key(j_node):
                trust_inverted[j_node][t] = node_trust_perspective[t][j_node]

    return trust_inverted


def generate_global_trust_values(trust_logs, metric_weights=None):
    """

    :param trust_logs:
    :param metric_weights:
    :return:
    """
    trust_perspectives = {
        node: generate_node_trust_perspective(node_observations, metric_weights=metric_weights)
        for node, node_observations in trust_logs.iteritems()
    }
    inverted_trust_perspectives = {
        node: invert_node_trust_perspective(node_perspective)
        for node, node_perspective in trust_perspectives.iteritems()
    }
    return trust_perspectives, inverted_trust_perspectives


def generate_trust_logs_from_comms_logs(comms_logs):
    """
    Returns the global trust log as a dict of each nodes observations at each time of each other node

    i.e. trust is internally recorded by each node wrt each node [node][t]
    for god processing it's easier to deal with [t][node]

    :param comms_logs:
    :return: trust observations[observer][t][target]
    """
    obs = {}
    trust = {node: log['trust'] for node, log in comms_logs.items()}
    for i_node, i_t in trust.items():
        # first pass to invert the observations
        if not obs.has_key(i_node):
            obs[i_node] = []
        for j_node, j_t in i_t.items():
            for o, observation in enumerate(j_t):
                while len(obs[i_node]) <= o:
                    obs[i_node].append({})
                obs[i_node][o][j_node] = observation
    return obs


def explode_metrics_from_trust_log(df, metrics_string=None):
    """
    This method presents an exploded view of the trust log where the individual metrics are column-wise with the
    per-node indexes shifted from the col space to the row-multiindex space

    :param metrics_string:
    tldr: turns the list-oriented value space in trust logs into a columular format.
    :param df:
    :return tf:
    """
    tf = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in df.stack().iterkv()}, orient='index')
    if metrics_string is None:
        metrics_string = "ATXP,ARXP,ADelay,ALength,Throughput,PLR"
    tf.columns = [metrics_string.split(',')]
    tf.index = pd.MultiIndex.from_tuples(tf.index, names=['var', 'run', 'observer', 't', 'target'])
    tf.index = tf.index.set_levels([
        tf.index.levels[0].astype(np.float64),
        tf.index.levels[1].astype(np.int32),
        tf.index.levels[2],
        tf.index.levels[3].astype(np.int32),
        tf.index.levels[4]
    ])
    tf.sort(inplace=True)
    return tf


def network_trust_dict(trust_run, observer='n0', recommendation_nodes=None, target='n1', indirect_nodes=None):
    """
    Take an individual simulation run and get a dict of the standard network perspectives across given recommenders and indirect nodes
    (you could probably cludge together a few runs and the data format would still be ok, but I wouldn't try plotting it directly)
    :param trust_run:
    :param observer:
    :param recommendation_nodes:
    :param target:
    :param indirect_nodes:
    :return:
    """
    if not recommendation_nodes:
        recommendation_nodes = ['n2', 'n3']
    if not indirect_nodes:
        indirect_nodes = ['n4', 'n5']

    t_whitenized = lambda x: max(_gray_whitenized(x)) * x  # Maps to max_s{f_s(T_{Bi})})T_{Bi}
    t_direct = lambda x: 0.5 * t_whitenized(x)
    t_recommend = lambda x: 0.5 * (
        2 * len(recommendation_nodes)
        / (2.0 * len(recommendation_nodes) + len(indirect_nodes))) * t_whitenized(x)
    t_indirect = lambda x: 0.5 * (
        len(indirect_nodes)
        / (2.0 * len(recommendation_nodes) + len(indirect_nodes))) * t_whitenized(x)

    network_list = [observer] + recommendation_nodes + indirect_nodes

    t_avg = trust_run.unstack('observer').xs(target, level='target', axis=1)[network_list].mean(axis=1)
    t_network = trust_run.unstack('observer').xs(target, level='target', axis=1)[network_list].applymap(t_whitenized).mean(axis=1)
    t_direct = trust_run.xs('n0', level='observer')['n1'].apply(t_direct)
    t_recommend = trust_run.unstack('observer').xs(target, level='target', axis=1)[recommendation_nodes].applymap(t_recommend).mean(axis=1)
    t_indirect = trust_run.unstack('observer').xs(target, level='target', axis=1)[indirect_nodes].applymap(t_indirect).mean(axis=1)

    t_total = pd.DataFrame.from_dict({
        'Direct': t_direct,
        'Recommend': t_recommend,
        'Indirect': t_indirect
    })

    # The driving philosophy of the following apparrant mess is that explicit is better that implicit;
    # If I screw up the data structure later; pandas will not forgive me.

    _d = pd.DataFrame.from_dict(OrderedDict((
        ("t10", trust_run.xs(observer, level='observer')[target]),
        ("t12", trust_run.xs('n2', level='observer')[target]),
        ("t13", trust_run.xs('n3', level='observer')[target]),
        ("t14", trust_run.xs('n4', level='observer')[target]),
        ("t15", trust_run.xs('n5', level='observer')[target]),
        ("t1-route_net", t_total.sum(axis=1)),  # Eq 4.7 guo; Takes relationships into account
        ("t1-white_net", pd.Series(t_network)),  # Eq 4.8 guo: Blind Whitenised Trust
        ("t1-avg", pd.Series(t_avg))  # Simple Average
    )))
    assert any(_d > 1), "All Resultantant Trust Values should be less than 1"
    assert any(0 > _d), "All Resultantant Trust Values should be greater than 0"
    return _d
