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

import itertools
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aietes import Tools
from bounos.Analyses import scenario_map



# USUALLY CONSTANTS
trust_metrics = np.asarray("ADelay,ARXP,ATXP,RXThroughput,PLR,TXThroughput".split(','))
metric_combinations = itertools.product(xrange(1, 4), repeat=len(trust_metrics))
metric_combinations_series = [pd.Series(x, index=trust_metrics) for x in metric_combinations]

# DEFAULT VARIABLES
default_mtfm_args = ('n0', 'n1', ['n2', 'n3'], ['n4', 'n5'])


def trust_from_file(file):
    with pd.get_store(Tools.in_results(file)) as store:
        trust = store.get('trust')
        Tools.map_levels(trust, scenario_map)
    return trust


def perspective_from_trust(trust, s=None, metric_weight=None, par=True, flip_metrics=None):
    if s is not None:
        trust = trust.xs(scenario_map[s], level='var')
    tp = generate_node_trust_perspective(trust, metric_weights=metric_weight, flip_metrics=flip_metrics, par=par)
    Tools.map_levels(tp, scenario_map)
    return tp


def generate_outlier_frame(mtfm, good_key, sigma=1.0):
    _mean = mtfm.xs(good_key, level='bev').mean()
    _std = mtfm.xs(good_key, level='bev').std()
    llim = _mean - (sigma * _std)
    ulim = _mean + (sigma * _std)
    outliers = pd.concat(
        (mtfm[mtfm > ulim] - ulim, (llim - mtfm[mtfm < llim])),
        keys=('upper', 'lower'), names=['range', 'bev', 't']
    )
    return outliers


def mtfm_from_perspectives_dict(perspectives, mtfm_args=None):
    if mtfm_args is None:
        mtfm_args = default_mtfm_args
    inter = pd.concat(perspectives.values(),
        axis=0, keys=perspectives.keys(),
        names=["bev"] + perspectives.values()[0].index.names)
    mtfms = (inter.groupby(level=['bev']) \
             .apply(generate_mtfm, *mtfm_args) \
             .reset_index(level=[0, 2, 3], drop=True) \
             .sum(axis=1)
             )
    return mtfms


def outliers_from_trust_dict(trust_dict, good_key="good", s=None,
                             metric_weight=None, mtfm_args=None, par=True,
                             flip_metrics=None):
    perspectives = {
        k: perspective_from_trust(t, s=s, metric_weight=metric_weight,
                                  flip_metrics=flip_metrics, par=par)
        for k, t in trust_dict.iteritems()
    }
    mtfms = mtfm_from_perspectives_dict(perspectives, mtfm_args)
    outlier = generate_outlier_frame(mtfms, good_key).reset_index()
    for k in metric_weight.keys():
        outlier[k] = metric_weight[k]
    outlier.set_index(['bev', 't'], inplace=True)
    outlier.rename(columns={0: 'Delta'}, inplace=True)
    return outlier

# TODO Grey stuff can probably be broken out at a separate file
# THESE LAMBDAS PERFORM GRAY CLASSIFICATION BASED ON GUO
# Still have no idea where sigma comes into it but that's life
_white_fs = [lambda x: -x + 1,
             lambda x: -2 * x + 2 if x > 0.5 else 2 * x,
             lambda x: x
             ]
_white_sigmas = [0.0, 0.5, 1.0]


def _gray_whitenized(x):
    return [f(x) for f in _white_fs]


def _t_whitenized(x):
    return max(_gray_whitenized(x)) * x  # Maps to max_s{f_s(T_{Bi})})T_{Bi}


def _t_direct(x):
    return 0.5 * _t_whitenized(x)


def _t_recommend(x, nr=np.inf, ni=np.inf):
    return 0.5 * ((2.0 * nr) / (2.0 * nr + ni)) * _t_whitenized(x)


def _t_indirect(x, nr=np.inf, ni=np.inf):
    return 0.5 * (ni / (2.0 * nr + ni)) * _t_whitenized(x)


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


def _grc(value, comparison, width, rho=0.5):
    """
    Generates the inner sequence for GRC
    Best / Worst based purely on comparison vector (i.e. good generally min(vals))
    Also applies vector scaling to [0,1]

    Validated against the code used in Guo (pg 183)

    :param value:
    :param comparison:
    :param width:
    :param rho:
    :return:
    """

    # todo test this against: zero widths, nan widths, singular arrays (should be 0.5)

    upper = width
    lower = np.abs(value - comparison) + rho * width
    with np.errstate(invalid='ignore'):
        # inner is in the range [2/3, 2]
        inner = upper / lower

    # Scale to [0,1]
    scaled = 0.75 * inner - 0.5

    return scaled


def generate_single_observer_trust_perspective(gf, metric_weights=None, flip_metrics=None, rho=0.5, debug=False):
    """
    Generate an individual observer perspective i.e per node record
    :param gf:
    :param metric_weights:
    :param flip_metrics: Metrics that are more 'good' as they get bigger
    :return:
    """
    trusts = []
    if debug:
        intervals = []
        goods = []
        bads = []
        keys = []
        maxes = []
        mins = []

    if flip_metrics is None:
        flip_metrics = ['TXThroughput', 'RXThroughput']  # These are 'bigger is better' values

    for ki, gi in gf.groupby(level='t'):
        gmn = gi.min(axis=0)  # Generally the 'Good' sequence,
        gmx = gi.max(axis=0)
        width = np.abs(gmx - gmn)

        # If we have any actual values
        if np.any(width[~np.isnan(width)] > 0):
            # While this looks bananas it is EXACTLY how it is in Bellas code.
            g_grc = partial(_grc, comparison=gmn, width=width)
            b_grc = partial(_grc, comparison=gmx, width=width)

            # Where there are 'missing' values, this implies that there's no information
            # therefore we can assume regression to the mean
            # This means that low-successful-metric records don't get too washed out by the limited info
            # EG if there is only TX Throughput, results are often 0 or 1 which makes v spiky
            good = gi.apply(g_grc, axis=1).fillna(0.5)
            bad = gi.apply(b_grc, axis=1).fillna(0.5)

            for flipper in flip_metrics:  # NOTE flipper may have been removed if no variation
                if flipper in good.keys() and flipper in bad.keys():
                    good[flipper], bad[flipper] = bad[flipper], good[flipper]

            if metric_weights is not None:
                # If the dropnas above have eliminated uninformative rows, they may have been trying to
                # be weighted on.... Fix that.
                # ALSO doing it in the same name is really bad for looping....
                valid_metric_weights = metric_weights.drop(metric_weights.keys().difference(good.keys()))
            else:
                valid_metric_weights = None

            try:
                interval = pd.DataFrame.from_dict({
                    'good': good.apply(np.average, weights=valid_metric_weights, axis=1),
                    'bad': bad.apply(np.average, weights=valid_metric_weights, axis=1)
                })[['good', 'bad']]
                t_val = pd.Series(
                    interval.apply(
                        t_kt,
                        axis=1),
                    name='trust'
                )
            except ValueError:
                print "Interval {}".format(interval)
                print "Good.keys(){}".format(good.keys())
                print "Bad.keys(){}".format(bad.keys())
                print "Orig Weight {}".format(metric_weights)
                print "New Weight {}".format(valid_metric_weights)
                print "Width {}".format(width)
                print "GI {}".format(gi)
                print "Good {}".format(good)
                print "Bad {}".format(bad)
                raise

            trusts.append(
                t_val
            )
        else:
            # If we don't have any records, there's nothing we can do.
            trusts.append(pd.Series([], name='trust'))
        if debug:
            interval.append(interval)
            goods.append(good)
            bads.append(bad)
            keys.append(ki)
            maxes.append(gmx)
            mins.append(gmn)
    if debug:
        debugging = {'intervals': intervals,
                     'goods': goods,
                     'bads': bads,
                     'keys': keys,
                     'maxes': maxes,
                     'mins': mins
                     }
        return trusts, debugging
    else:
        return trusts


def generate_node_trust_perspective(tf, var='var', metric_weights=None, flip_metrics=None,
                                    rho=0.5, fillna=False, par=True):
    """
    Generate Trust Values based on a big trust_log frame (as acquired from multi_loader or from explode_metrics_...
    Will also accept a selectively filtered trust log for an individual run
    i.e node_trust where node_trust is the inner result of:
        trust.groupby(level=['var','run','node'])


    Parallel Performs significantly better on large(r) datasets, i.e. multi-var.
    ie. Linear- ~2:20s/run vs 50s/run parallel

    BY DEFAULT FLIPS THROUGHPUT METRICS

    :param tf: pandas.DataFrame: Trust Metrics DataFrame; can be single or ['var','run'] indexed
    :param var: str: optional level name to group by as opposed to the standard 'var'
    :param metric_weights: numpy.ndarray: per-metric weighting array (default None)
    :param n_metrics: int: number of metrics assessed in each observation
    :return: pandas.DataFrame
    """
    assert isinstance(tf, pd.DataFrame), "Expected first argument (tf) to be a Pandas Dataframe, got {} instead".format(
        type(tf)
    )

    if var not in tf.index.names and 'run' not in tf.index.names:
        # Dealing with a single run; pad it with 0's
        dff = pd.concat([tf], keys=[0] + tf.index.names, names=['run'] + tf.index.names)
        tf = pd.concat([dff], keys=[0] + dff.index.names, names=[var] + dff.index.names)
    elif var not in tf.index.names:
        tf = pd.concat([tf], keys=[0] + tf.index.names, names=[var] + tf.index.names)

    trusts = []

    exec_args = {'metric_weights': metric_weights, 'flip_metrics': flip_metrics, 'rho': rho}

    if par:
        trusts = Parallel(n_jobs=-1)(delayed(generate_single_observer_trust_perspective)
                                     (g, **exec_args) for k, g in tf.groupby(level=[var, 'run', 'observer'])
        )
        trusts = [item for sublist in trusts for item in sublist]
    else:
        with np.errstate(all='raise'):
            for k, g in tf.groupby(level=[var, 'run', 'observer']):
                trusts.extend(generate_single_observer_trust_perspective(g, **exec_args))

    tf = pd.concat(trusts)
    tf.sort(inplace=True)

    # The following:
    # Transforms the target id into the column space,
    # Groups each nodes independent observations together
    # Fills in the gaps IN EACH ASSESSMENT with the previous assessment of that node by that node at the previous time
    if fillna:
        tf = tf.unstack('target').groupby(level=[var, 'run', 'observer']).apply(lambda x: x.fillna(method='ffill'))
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
    raise PendingDeprecationWarning("This shouldn't be used any more and will be removed soon")
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


def generate_mtfm(trust_run, observer, target, recommendation_nodes, indirect_nodes, ewma=False):
    # Perform MTMF
    t_recommend = partial(_t_recommend, nr=len(recommendation_nodes), ni=len(indirect_nodes))
    t_indirect = partial(_t_indirect, nr=len(recommendation_nodes), ni=len(indirect_nodes))
    t_direct_val = trust_run.xs(observer, level='observer')[target].apply(_t_direct)
    t_recommend_val = trust_run.unstack('observer').xs(target, level='target', axis=1)[recommendation_nodes].applymap(
        t_recommend).mean(axis=1)
    t_indirect_val = trust_run.unstack('observer').xs(target, level='target', axis=1)[indirect_nodes].applymap(
        t_indirect).mean(axis=1)
    t_mtfm_val = pd.DataFrame.from_dict({
        'Direct': t_direct_val,
        'Recommend': t_recommend_val,
        'Indirect': t_indirect_val
    })
    if ewma:
        t_mtfm_val = t_mtfm_val.groupby(level=['var', 'run']).apply(
            lambda s: pd.stats.moments.ewma(s.fillna(method="ffill"), span=16))
    return t_mtfm_val


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
    :return: tuple: (t_average, t_white, t_mtfm
    """
    trust_run = trust_run.unstack('observer').groupby(level=['var', 'run']).apply(
        lambda s: s.fillna(method='ffill')
    ).stack('observer')

    if recommendation_nodes is None:
        recommendation_nodes = ['n2', 'n3']
    if indirect_nodes is None:
        indirect_nodes = ['n4', 'n5']

    network_list = [observer] + recommendation_nodes + indirect_nodes

    t_avg = trust_run.unstack('observer').xs(target, level='target', axis=1)[network_list].mean(axis=1)
    t_network = trust_run.unstack('observer').xs(target, level='target', axis=1)[network_list].applymap(
        _t_whitenized).mean(axis=1)

    t_mtfm = generate_mtfm(trust_run, observer, target, recommendation_nodes, indirect_nodes, ewma)

    return t_avg, t_network, t_mtfm


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

    t_avg, t_network, t_total = generate_network_trust(trust_run, observer, recommendation_nodes,
                                                       target, indirect_nodes, ewma)

    # The driving philosophy of the following apparrant mess is that explicit is better that implicit;
    # If I screw up the data structure later; pandas will not forgive me.

    _d = pd.DataFrame.from_dict(OrderedDict((
        ("$T_{0}$", trust_run.xs(observer, level='observer')[target]),
        ("$T_{2}$", trust_run.xs('n2', level='observer')[target]),
        ("$T_{3}$", trust_run.xs('n3', level='observer')[target]),
        ("$T_{4}$", trust_run.xs('n4', level='observer')[target]),
        ("$T_{5}$", trust_run.xs('n5', level='observer')[target]),
        ("$T_{Net}$", t_total.sum(axis=1)),  # Eq 4.7 guo; Takes relationships into account
        ("$T_{MTFM}$", pd.Series(t_network)),  # Eq 4.8 guo: Blind Whitenised Trust
        ("$T_{Avg}$", pd.Series(t_avg))  # Simple Average
    )))

    # On combining assessments, Nans are added where different nodes have (no) information at a particular timeframe about
    # a particular target. Fix that.
    _d.fillna(method="ffill", inplace=True)

    assert any(_d > 1), "All Resultantant Trust Values should be less than 1"
    assert any(0 > _d), "All Resultantant Trust Values should be greater than 0"
    return _d
