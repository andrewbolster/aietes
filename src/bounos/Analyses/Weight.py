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


import pandas as pd
import numpy as np

import itertools
from functools import partial

from bounos.Analyses import Trust

from bounos.ChartBuilders import weight_comparisons


def generate_trust_metric_weights(trust_metric_names, exclude = None, max_emphasis = 5):
    """
    Generate trust metric weights from their base labels (eg PLR,Delay, etc)
    Creates normalised (sum==1) weight vectors for application to grey vectors

    :param trust_metric_names:
    :param exclude:
    :param max_emphasis: number of steps available for variation of emphasis
    :return:
    """
    if exclude is None:
        exclude = []

    trust_combinations = []
    map(trust_combinations.extend,
        np.asarray([itertools.combinations(trust_metric_names, i)
                    for i in range(max_emphasis, len(trust_metric_names))])
        )
    trust_combinations = np.asarray(filter(lambda x: all(map(lambda m: m not in exclude, x)), trust_combinations))
    trust_metric_selections = np.asarray(
        [map(lambda m: float(m in trust_combination), trust_metric_names) for trust_combination in trust_combinations])
    trust_metric_weights = map(lambda s: s / sum(s), trust_metric_selections)

    return trust_metric_weights



def generate_outlier_tp(tf, sigma=None, good='good', good_lvl='bev'):
    """
    This must be applied on a per-observation basis (i.e. single run, single observer)
    :param tf:
    :param sigma:
    :param good:
    :param good_lvl:
    :return:
    """
    if sigma is not None:
        raise NotImplementedError("!Have't implemented")

    _mean=tf.xs(good,level=good_lvl).mean()
    _std=tf.xs(good,level=good_lvl).std()
    llim=_mean-_std
    ulim=_mean+_std
    uppers = (tf[tf>ulim]-ulim).reset_index()
    lowers = (llim-tf[tf<llim]).reset_index()
    outliers = pd.concat((uppers,lowers), keys=('upper','lower'), names=['range','id']).reset_index().set_index(
        ['var','run','observer','range','t']
    )
    outliers.drop('id', axis=1, inplace=True)
    return outliers




def generate_outlier_frame(good, trust_frame, w, par=False):
    try:
        _ = trust_frame.xs(good, level='var').mean()
    except KeyError:
        raise ValueError("Need to set a good behaviour that exists in the trust_frame")

    # TODO experiment with wether it's better to parallelise the inner or outer version of this.... (i.e. par/w vs par/single
    weighted_trust_perspectives = Trust.generate_node_trust_perspective(trust_frame,
                                                                        metric_weights=w,
                                                                        par=par)
    l_outliers = []
    for i, tf in weighted_trust_perspectives.groupby(level=['run', 'observer']):
        l_outliers.append(generate_outlier_tp(tf, good=good, good_lvl='var'))
    outlier = pd.concat(l_outliers).sort_index().dropna(how='all').reset_index()
    for k in w.keys():
        outlier[k] = w[k]
    outlier.set_index(['var', 't'], inplace=True)
    return outlier
