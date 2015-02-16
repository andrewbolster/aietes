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


def generate_single_observer_trust_perspective(gf, metric_weights=None, flip_metrics=None, rho=0.5):
    """
    Generate an individual observer perspective i.e per node record
    :param gf:
    :param metric_weights:
    :param flip_metrics:
    :param rho:
    :return:
    """
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

    :param tf: Trust Metrics DataFrame; can be single or ['var','run'] indexed
    :param metric_weights: per-metric weighting array (default None)
    :param n_metrics: number of metrics assessed in each observation
    :return:
    """
    assert isinstance(tf, pd.DataFrame)

    if 'var' not in tf.index.names and 'run' not in tf.index.names:
        # Dealing with a single run; pad it with 0's
        dff = pd.concat([tf], keys=[0] + tf.index.names, names=['run'] + tf.index.names)
        tf = pd.concat([dff], keys=[0] + dff.index.names, names=['var'] + dff.index.names)

    trusts = []

    if flip_metrics is None:
        flip_metrics = ['ADelay', 'PLR']

    exec_args = {'metric_weights': metric_weights, 'flip_metrics': flip_metrics, 'rho': rho}

    if par:
        trusts = Parallel(n_jobs=-1)(delayed(generate_single_observer_trust_perspective)
                                     (g, **exec_args) for k, g in tf.groupby(level=['var', 'run', 'observer'])
        )
        trusts = [item for sublist in trusts for item in sublist]
    else:
        for k, g in tf.groupby(level=['var', 'run', 'observer']):
            trusts.extend(generate_single_observer_trust_perspective(g, **exec_args))

    tf = pd.concat(trusts)
    tf.sort(inplace=True)

    # The following:
    # Transforms the target id into the column space,
    # Groups each nodes independent observations together
    # Fills in the gaps IN EACH ASSESSMENT with the previous assessment of that node by that node at the previous time
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

    return pd.concat(
        [pd.concat(
            [pd.DataFrame(target_metrics) for target_metrics in observer_metrics.itervalues()],
            keys=observer_metrics.keys()) for observer_metrics in trust.itervalues()],
        keys=trust.keys(), names=['observer', 'target', 't'])


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
    tf.index = pd.MultiIndex.from_tuples(tf.index, names=['var', 'run', 'observer', 't', 'target'])
    try:
        map(float, df.index.levels[0])
        var_is_float = True
    except:
        var_is_float = False

    tf.index = tf.index.set_levels([
        tf.index.levels[0].astype(np.float64) if var_is_float else df.index.levels[0],  # Var
        tf.index.levels[1].astype(np.int32),  # run
        tf.index.levels[2],  # observer
        tf.index.levels[3].astype(np.int32),  # t
        tf.index.levels[4]  # target
    ])
    tf.sort(inplace=True)
    return tf


def generate_network_trust(trust_run, observer='n0', recommendation_nodes=None, target='n1',
                           indirect_nodes=None, ewma=True):
    """
    Perform a full networked trust assessment for given observers perspective with given
    recommendation and indirect nodes contributions.

    Also performs simple average and whitenized network

    :param trust_run:
    :param observer:
    :param recommendation_nodes:
    :param target:
    :param indirect_nodes:
    :param ewma:
    :return: tuple: (t_average, t_white, t_mtmf
    """
    trust_run = trust_run.unstack('observer').groupby(level=['var', 'run']).apply(
        lambda s: s.fillna(method='ffill')
    ).stack('observer')

    t_whitenized = lambda x: max(_gray_whitenized(x)) * x  # Maps to max_s{f_s(T_{Bi})})T_{Bi}
    t_direct = lambda x: 0.5 * t_whitenized(x)
    t_recommend = lambda x: 0.5 * (
        (2.0 * len(recommendation_nodes))
        / (2.0 * len(recommendation_nodes) + len(indirect_nodes))) * t_whitenized(x)
    t_indirect = lambda x: 0.5 * (
        float(len(indirect_nodes))
        / (2.0 * len(recommendation_nodes) + len(indirect_nodes))) * t_whitenized(x)

    network_list = [observer] + recommendation_nodes + indirect_nodes

    t_avg = trust_run.unstack('observer').xs(target, level='target', axis=1)[network_list].mean(axis=1)
    t_network = trust_run.unstack('observer').xs(target, level='target', axis=1)[network_list].applymap(
        t_whitenized).mean(axis=1)

    # Perform MTMF
    t_direct = trust_run.xs('n0', level='observer')['n1'].apply(t_direct)
    t_recommend = trust_run.unstack('observer').xs(target, level='target', axis=1)[recommendation_nodes].applymap(
        t_recommend).mean(axis=1)
    t_indirect = trust_run.unstack('observer').xs(target, level='target', axis=1)[indirect_nodes].applymap(
        t_indirect).mean(axis=1)
    t_mtmf = pd.DataFrame.from_dict({
        'Direct': t_direct,
        'Recommend': t_recommend,
        'Indirect': t_indirect
    })

    if ewma:
        t_mtmf = t_mtmf.groupby(level=['var', 'run']).apply(
            lambda s: pd.stats.moments.ewma(s.fillna(method="ffill"), span=16))

    return t_avg, t_network, t_mtmf


def network_trust_dict(trust_run, observer='n0', recommendation_nodes=None, target='n1',
                       indirect_nodes=None, ewma=False):
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

    t_avg, t_network, t_total = generate_network_trust(trust_run, observer, recommendation_nodes,
                                                       target, indirect_nodes, ewma)

    # The driving philosophy of the following apparrant mess is that explicit is better that implicit;
    # If I screw up the data structure later; pandas will not forgive me.

    _d = pd.DataFrame.from_dict(OrderedDict((
        ("$T_{1,0}$", trust_run.xs(observer, level='observer')[target]),
        ("$T_{1,2}$", trust_run.xs('n2', level='observer')[target]),
        ("$T_{1,3}$", trust_run.xs('n3', level='observer')[target]),
        ("$T_{1,4}$", trust_run.xs('n4', level='observer')[target]),
        ("$T_{1,5}$", trust_run.xs('n5', level='observer')[target]),
        ("$T_{1,Net}$", t_total.sum(axis=1)),  # Eq 4.7 guo; Takes relationships into account
        ("$T1_{1,MTFM}$", pd.Series(t_network)),  # Eq 4.8 guo: Blind Whitenised Trust
        ("$T1_{1,Avg}$", pd.Series(t_avg))  # Simple Average
    )))

    # On combining assessments, Nans are added where different nodes have (no) information at a particular timeframe about
    # a particular target. Fix that.
    _d.fillna(method="ffill", inplace=True)

    assert any(_d > 1), "All Resultantant Trust Values should be less than 1"
    assert any(0 > _d), "All Resultantant Trust Values should be greater than 0"
    return _d
