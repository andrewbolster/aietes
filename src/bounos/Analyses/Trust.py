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
    return np.average(c_intervals, axis=1, weights=None)

def T_kt(interval):
    """
    Generate a single trust value from a GRG
    1/ (1+ sigma^2/theta^2)
    :param interval:
    :return:
    """
    theta, sigma = interval
    return 1.0 / (
        1.0 + (
            np.power(sigma, 2) / np.power(theta, 2)
        )
    )

def generate_node_trust_perspective(node_observations, metric_weights=None, n_metrics=6):
    """
    Generate Trust Values based on each nodes trust log (dp.get_global_trust_logs[observer])
    :param node_observations: node observations [t][target][x,x,x,x,x,x]
    :param metric_weights: per-metric weighting array (default None)
    :param n_metrics: number of metrics assessed in each observation
    :return:
    """
    trust=[]
    grc=grc_factory(0.5)
    for t, t_obs in enumerate(node_observations):
        #Sweep across the nodes observed in this time and take the g/b
        # indexes
        g=np.array([np.inf for _ in range(n_metrics)])
        b=np.zeros_like(g)
        for j_node, j_obs in t_obs.items():
            if len(j_obs):
                g=np.min([j_obs,g], axis=0)
                b=np.max([j_obs,b], axis=0)
        # Now that we have the best reference sequences

        # Inherit lasst trust values for missing trusts
        if not t:
            td={}
        else:
            td=deepcopy(trust[-1])

        # Perform Grey Relational Trust Calculation
        for j_node, j_obs in t_obs.items():
            if len(j_obs):
                t_val = T_kt(
                    GRG_t(
                        np.asarray(map(grc, [j_obs - g, j_obs - b])),
                        weights=metric_weights)
                )
                if not np.isnan(t_val):
                    td[j_node]=t_val
        trust.append(td)
    return trust

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

def generate_global_trust_values(dp):
    trust_perspectives={
        node: generate_node_trust_perspective(node_observations)
        for node, node_observations in dp.get_global_trust_logs().items()
    }
    inverted_trust_perspectives={
        node: invert_node_trust_perspective(node_perspective)
        for node, node_perspective in trust_perspectives.items()
    }
    return trust_perspectives, inverted_trust_perspectives

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
