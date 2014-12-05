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
n_metrics = 6

import numpy as np
import pandas as pd

from copy import deepcopy

# THESE LAMBDAS PERFORM GRAY CLASSIFICATION BASED ON GUO
# Still have no idea where sigma comes into it but that's life
_white_fs=[lambda x : -x + 1,
           lambda x : -2*x +2 if x>0.5 else 2*x,
           lambda x : x
]
_white_sigmas=[0.0,0.5,1.0]

_gray_whitenized=lambda x : map(lambda f: f(x), _white_fs)
_gray_class = lambda x: (_gray_whitenized(x).index(max(_gray_whitenized(x))))


def grc_factory(rho=0.5):
    """
    Factory function for generating Grey Relational Coeffiecient functions
        based on the distringuishing coefficient (rho)
    :param rho: float, distinguishing coefficient (rho)
    :return grc: function
    """
    def grc(delta):
        """
        GRC Inner function
        Includes remapping of GRC to 0<x<1 rather than 1/3<x<1
        :param delta:
        :param rho:
        :return:
        """
        delta = abs(delta)
        upper = np.min(delta, axis=0) + (rho * np.max(delta, axis=0))
        lower = (delta) + (rho * np.max(delta, axis=0))
        with np.errstate(invalid='ignore', ):
            parterval = np.divide(upper, lower)
        return 1.5*parterval-0.5

    return grc

def GRG_t(c_intervals, weights=None):
    """
    Grey Relational Grade

    Weighted sum given input structure

        [node,metric,[interval]]

    returns
        [node, [interval]]
    :param grcs:
    :return:
    """
    return np.average(c_intervals, axis=1, weights=weights)

def T_kt(interval):
    """
    Generate a single trust value from a GRG
    1/ (1+ sigma^2/theta^2)
    :param interval:
    :return:
    """
    # good, bad
    theta, sigma = interval
    return 1.0 / (
        1.0 + (sigma*sigma) / (theta*theta)
    )

def generate_node_trust_perspective(node_observations, metric_weights=None, n_metrics=6):
    """
    Generate Trust Values based on each nodes trust log (dp.get_global_trust_logs[observer])
    Will also accept a selectively filtered trust log for an individual run
    i.e node_trust where node_trust is the inner result of:
        trust.groupby(level=['var','run','node'])
    :param node_observations: node observations [t][target][x,x,x,x,x,x]
    :param metric_weights: per-metric weighting array (default None)
    :param n_metrics: number of metrics assessed in each observation
    :return:
    """
    raise PendingDeprecationWarning("This function doesn't implement metric inversion and should not be used anymore")
    def strip_leading_iterators(tup):
        for (_,_,_,t),t_obs in tup():
            yield t,t_obs

    trust=[]
    grc=grc_factory(0.5)
    if not isinstance(node_observations,list):
        assert isinstance(node_observations, pd.DataFrame)
        node_obs_gen = strip_leading_iterators(node_observations.dropna(axis=1).iterrows)
    elif isinstance(node_observations,list):
        node_obs_gen = enumerate(node_observations)
    for t, t_obs in node_obs_gen:
        #Sweep across the nodes observed in this time and take the g/b
        # indexes
        g=np.array([np.inf for _ in range(n_metrics)])
        b=np.zeros_like(g)
        try:
            for j_node, j_obs in t_obs.iteritems():
                if len(j_obs):
                    g=np.min([j_obs,g], axis=0)
                    b=np.max([j_obs,b], axis=0)
        except:
            raise
        # Now that we have the best reference sequences

        # Inherit lasst trust values for missing trusts
        if not t:
            td={}
        else:
            td=deepcopy(trust[-1])

        # Perform Grey Relational Trust Calculation
        for j_node, j_obs in t_obs.iteritems():
            if len(j_obs):
                t_val = T_kt(
                    GRG_t(
                        map(grc, [j_obs - g, j_obs - b]),
                        weights=metric_weights)
                )
                if not np.isnan(t_val):
                    td[j_node]=t_val
        trust.append(td)
    return trust


def generate_node_trust_perspective_from_trust_frame(tf, metric_weights=None, n_metrics=6):
    """
    Generate Trust Values based on a big trust_log frame (as acquired from multi_loader or from explode_metrics_...
    Will also accept a selectively filtered trust log for an individual run
    i.e node_trust where node_trust is the inner result of:
        trust.groupby(level=['var','run','node'])
    :param node_observations: node observations [t][target][x,x,x,x,x,x]
    :param metric_weights: per-metric weighting array (default None)
    :param n_metrics: number of metrics assessed in each observation
    :return:
    """
    assert isinstance(tf, pd.DataFrame)
    trusts={}
    grc=grc_factory(0.5)
    for k,g in tf.dropna().groupby(level=['var','run','observer']):
        for ki, gi in g.groupby(level='t'):
            gg=gi.min()
            gb=gi.max()
            for n,o in gi.iterrows():
                trusts[n]=T_kt(
                    GRG_t(
                        np.asarray(map(grc,[o-gg,o-gb])),
                        weights=metric_weights)
                )

    tf=pd.DataFrame.from_dict(trusts, orient='index')
    tf.index = pd.MultiIndex.from_tuples(tf.index, names=['var','run','observer','t','target'])
    tf.index=tf.index.set_levels([
        tf.index.levels[0].astype(np.float64),#Var
        tf.index.levels[1].astype(np.int32),#Run
        tf.index.levels[2],#Node
        tf.index.levels[3].astype(np.int32),#Target (should really be a time)
        tf.index.levels[4]  #Target
    ])
    tf.sort(inplace=True)

    # The following:
    #   Transforms the target id into the column space,
    #   Groups each nodes independent observations together
    #   Fills in the gaps IN EACH ASSESSMENT with the previous assessment of that node by that node at the previous time

    tf = tf.unstack('target').groupby(level=['var','run','observer']).apply(lambda x: x.fillna(method='ffill'))
    return tf

def generate_node_trust_perspective_from_trust_frame_apply(tf, metric_weights=None, flip_metrics=None, rho=0.5):

    """
    Generate Trust Values based on a big trust_log frame (as acquired from multi_loader or from explode_metrics_...
    Will also accept a selectively filtered trust log for an individual run
    i.e node_trust where node_trust is the inner result of:
        trust.groupby(level=['var','run','node'])
    :param node_observations: node observations [t][target][x,x,x,x,x,x]
    :param metric_weights: per-metric weighting array (default None)
    :param n_metrics: number of metrics assessed in each observation
    :return:
    """
    assert isinstance(tf, pd.DataFrame)
    grc=grc_factory(0.5)
    trusts=[]
    if flip_metrics is None:
        flip_metrics = ['ADelay','PLR']
    for k,g in tf.dropna().groupby(level=['var','run','observer']):
        for ki, gi in g.groupby(level='t'):
            gmx=gi.max()
            gmn=gi.min()
            width=gmx-gmn

            good=gi.apply(
                lambda o:(0.75*np.divide((width),(np.abs(o-gmn))+rho*(width))-0.5).fillna(1),
                axis=1
            )
            bad=gi.apply(
                lambda o:(0.75*np.divide((width),(np.abs(o-gmx))+rho*(width))-0.5).fillna(1),
                axis=1
            )

            good[flip_metrics],bad[flip_metrics]=bad[flip_metrics],good[flip_metrics]

            interval=pd.DataFrame.from_dict({
            'good': good.apply(np.average, weights=metric_weights, axis=1),
            'bad': bad.apply(np.average, weights=metric_weights, axis=1)
            })
            trusts.append(
                pd.concat(
                    [gi,
                     interval,
                     pd.Series(
                         interval.apply(
                             lambda o:1/(1+((o[1]*o[1])/(o[0]*o[0]))),
                             axis=1),
                         name='trust')
                    ],
                    axis=1)
            )

    tf=pd.concat(trusts)
    tf.index = pd.MultiIndex.from_tuples(tf.index, names=['var','run','observer','t','target'])
    tf.index=tf.index.set_levels([
        tf.index.levels[0].astype(np.float64),#Var
        tf.index.levels[1].astype(np.int32),#Run
        tf.index.levels[2],#Node
        tf.index.levels[3].astype(np.int32),#Target (should really be a time)
        tf.index.levels[4]  #Target
    ])
    tf.sort(inplace=True)

    # The following:
    #   Transforms the target id into the column space,
    #   Groups each nodes independent observations together
    #   Fills in the gaps IN EACH ASSESSMENT with the previous assessment of that node by that node at the previous time

    #tf = tf.unstack('target').groupby(level=['var','run','observer']).apply(lambda x: x.fillna(method='ffill'))
    return tf

def invert_node_trust_perspective(node_trust_perspective):
    """
    Invert Node Trust Records to unify against time, i.e. [observer][t][target]
    :param node_trust_perspective:
    :return:
    """
    # trust[observer][t][target] = T_jkt
    trust_inverted={}
    for j_node in node_trust_perspective[-1].keys():
        trust_inverted[j_node]=np.array([0.5 for _ in range(len(node_trust_perspective))])
        for t in range(len(node_trust_perspective)):
            if t<len(node_trust_perspective) and node_trust_perspective[t].has_key(j_node):
                trust_inverted[j_node][t]=node_trust_perspective[t][j_node]

    return trust_inverted

def generate_global_trust_values(trust_logs, metric_weights=None):
    trust_perspectives={
        node: generate_node_trust_perspective(node_observations, metric_weights=metric_weights)
        for node, node_observations in trust_logs.iteritems()
    }
    inverted_trust_perspectives={
        node: invert_node_trust_perspective(node_perspective)
        for node, node_perspective in trust_perspectives.iteritems()
    }
    return trust_perspectives, inverted_trust_perspectives

def generate_trust_logs_from_comms_logs(comms_logs):
    """
    Returns the global trust log as a dict of each nodes observations at each time of each other node

    i.e. trust is internally recorded by each node wrt each node [node][t]
    for god processing it's easier to deal with [t][node]

    :return: trust observations[observer][t][target]
    """
    obs={}
    trust = { node: log['trust'] for node, log in comms_logs.items()}
    for i_node, i_t in trust.items():
        # first pass to invert the observations
        if not obs.has_key(i_node):
            obs[i_node]=[]
        for j_node, j_t in i_t.items():
            for o, observation in enumerate(j_t):
                while len(obs[i_node])<=(o):
                    obs[i_node].append({})
                obs[i_node][o][j_node]=observation
    return obs

def generate_trust_values_from_trust_log(df, metric_weights=None, n_metrics=6):
    """
    Given a trust log dataframe, return the metric weighted Gray Theoretic trust perspectives as a dataframe
    :param df:
    :param metric_weights:
    :param n_metrics:
    :return:
    """
    _temp_frame = df.groupby(level=['var','run','observer']).apply(lambda x: generate_node_trust_perspective(x, metric_weights=metric_weights, n_metrics=n_metrics))
    trust_frame = pd.concat(
        # Join all Runs into a Single Var
        [pd.concat(
            # Join all perspectives into a single run
            [pd.DataFrame(trusts) for trusts in val],
            keys=val.keys()
        ) for grp, val in _temp_frame.groupby(level=['var','run'])]
    )
    trust_frame.index.names=['var','run','observer','t']
    trust_frame.columns.names=['target']
    return trust_frame

def explode_metrics_from_trust_log(df, metrics_string=None):
    """
    This method presents an exploded view of the trust log where the individual metrics are column-wise with the
    per-node indexes shifted from the col space to the row-multiindex space

    tldr: turns the list-oriented value space in trust logs into a columular format.
    :param df:
    :return tf:
    """
    tf=pd.DataFrame.from_dict({k:v for k,v in df.iterkv()}, orient='index')
    if metrics_string is None:
        metrics_string="ATXP,ARXP,ADelay,ALength,Throughput,PLR"
    tf.columns=[metrics_string.split(',')]
    tf.index = pd.MultiIndex.from_tuples(tf.index, names=['var','run','observer','t','target'])
    tf.index=tf.index.set_levels([
        tf.index.levels[0].astype(np.float64),
        tf.index.levels[1].astype(np.int32),
        tf.index.levels[2],
        tf.index.levels[3].astype(np.int32),
        tf.index.levels[4]
    ])
    tf.sort(inplace=True)
    return tf


def network_trust_dict(trust_run, observer='n0', recommendation_nodes = ['n2','n3'], target = 'n1', indirect_nodes = ['n4','n5']):
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
    t_direct = lambda x: 0.5 * max(_gray_whitenized(x)) * x
    t_recommend = lambda x: 0.5 * (\
            2*len(recommendation_nodes) \
            /(2.0*len(recommendation_nodes)+len(indirect_nodes))) * max(_gray_whitenized(x)) * x
    t_indirect = lambda x: 0.5 * (\
            2*len(indirect_nodes) \
            /(2.0*len(recommendation_nodes)+len(indirect_nodes))) * max(_gray_whitenized(x)) * x

    def total_trust(t):
        Td = t_direct(trust_run[observer][target][t])
        Tr = np.average([t_recommend(trust_run[recommender][target][t]) for recommender in recommendation_nodes])
        Ti = np.average([t_indirect(trust_run[indirecter][target][t]) for indirecter in indirect_nodes])
        return sum((Td,Tr,Ti))

    network_list = [observer] + recommendation_nodes + indirect_nodes

    T_network = trust_run.unstack('observer').xs(target, level='target',axis=1)[network_list].mean(axis=1)
    T_direct = trust_run.xs('n0', level='observer')['n1'].apply(t_direct)
    T_recommend= trust_run.unstack('observer').xs(target, level='target',axis=1)[recommendation_nodes].applymap(t_recommend).mean(axis=1)
    T_indirect = trust_run.unstack('observer').xs(target, level='target',axis=1)[indirect_nodes].applymap(t_indirect).mean(axis=1)
    T_total=pd.DataFrame.from_dict({
        'Direct': T_direct,
        'Recommend': T_recommend,
        'Indirect': T_indirect
    })


    # The driving philosophy of the following apparrant mess is that explicit is better that implicit;
    # If I screw up the data structure later; pandas will not forgive me.

    _d=pd.DataFrame.from_dict(
    {"t10": trust_run.xs(observer, level='observer')[target],
    "t12": trust_run.xs('n2', level='observer')[target],
    "t13": trust_run.xs('n3', level='observer')[target],
    "t14": trust_run.xs('n4', level='observer')[target],
    "t15": trust_run.xs('n5', level='observer')[target],
    "t10-5": T_total.sum(axis=1),
    "t10-net": pd.Series(T_network)
    })
    return _d

def dev_to_trust(per_metric_deviations):
    # rotate pmdev to node-primary ([node,metric])
    per_node_deviations = np.rollaxis(per_metric_deviations, 0, 3)
    GRC_t = grc_factory()
    grcs = map(GRC_t, per_node_deviations)
    grgs = map(GRG_t, grcs)
    trust_values = np.asarray([
        np.asarray([
            T_kt(interval) for interval in node
        ]) for node in grgs
    ])
    return trust_values
