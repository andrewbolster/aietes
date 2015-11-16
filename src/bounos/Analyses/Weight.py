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
import sklearn.ensemble as ske

import numpy as np
from time import gmtime, strftime


import itertools
from joblib import Parallel, delayed

from functools import partial

from bounos.Analyses import Trust

from bounos.ChartBuilders import weight_comparisons
from bounos.ChartBuilders.weight_comparisons import norm_weight



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
                                                                        flip_metrics=[],
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

def perform_weight_factor_analysis_on_trust_frame(trust_frame, good, min_emphasis = 0, max_emphasis = 3, extra=None, verbose=False, par=False):
    """
    For a given trust frame perform grey weight factor distribution analysis to
    generate a frame with metrics/metric weights as a multiindex with columns for each comparison between behaviours

    :param trust_frame:
    :param good: the baseline behaviour to assess against
    :param min_metrics:
    :param: max_metrics:
    :param: max_emphasis:
    :return:
    """
    # Extract trust metric names from frame
    trust_metrics = list(trust_frame.keys())
    if max_emphasis - min_emphasis < 2:
        raise RuntimeError("Setting Max Emphasis <2 is pointless; Runs all Zeros")

    if extra is None:
        extra = ""

    combinations = itertools.product(xrange(min_emphasis, max_emphasis), repeat=len(trust_metrics))
    if par:
        outliers = _outlier_par_inner_single_thread(combinations, good, trust_frame, trust_metrics, verbose=True)
    else:
        outliers = _outlier_single_thread_inner_par(combinations, good, trust_frame, trust_metrics, verbose=True)

    sums = pd.concat(outliers).reset_index()
    sums.to_hdf('/home/bolster/src/aietes/results/outlier_backup.h5', "{}{}_{}".format(good,extra,max_emphasis))
    return sums


def _outlier_single_thread_inner_par(combinations, good, trust_frame, trust_metrics, verbose):
    outliers = []
    for i, w in enumerate(combinations):
        if verbose: print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), i, w)
        outlier = generate_outlier_frame(good, trust_frame, norm_weight(w, trust_metrics), par=True)
        outliers.append(outlier)
    return outliers

def _outlier_par_inner_single_thread(combinations, good, trust_frame, trust_metrics, verbose):

    outliers = Parallel(n_jobs=-1, verbose=int(verbose)*50)(delayed(generate_outlier_frame)
                                   (good, trust_frame, norm_weight(w, trust_metrics), False)
                                   for w in combinations)
    return outliers


def feature_extractor(df, target):
    data = df.drop(target, axis=1)
    reg = ske.RandomForestRegressor(n_jobs=4, n_estimators=512)
    reg.fit(data, df[target])
    return pd.Series(dict(zip(data.keys(), reg.feature_importances_)))

# The questions we want to answer are:
# 1. What metrics differentiate between what behaviours
# 2. Do these metrics cross over domains (i.e. comms impacting behaviour etc)
#
# To answer these questions we first have to manipulate the raw dataframe to be weight-indexed with behaviour(`var`) keys on the perspective from the observer to the (potential) attacker in a particular run (summed across the timespace)
#
# While this analysis simply sums both the upper and lower outliers, **this needs extended/readdressed**
#
# # IMPORTANT
# The July 3rd Simulation Run had a small mistake where the Ran

# In[10]:

def categorise_dataframe(df):
    # Categories work better as indexes
    for obj_key in df.keys()[df.dtypes == object]:
        try:
            df[obj_key] = df[obj_key].astype('category')
        except TypeError:
            print("Couldn't categorise {}".format(obj_key))
            pass
    return df


def summed_outliers_per_weight(weight_df, observer, n_metrics, target=None):

    # Select Perspective here
    weight_df = weight_df[weight_df.observer == observer]
    weight_df = categorise_dataframe(weight_df)

    # Metrics are the last sector of the frame, set these as leading indices
    metric_keys = list(weight_df.keys()[-n_metrics:])
    weight_df.set_index(metric_keys + ['var', 't'], inplace=True)

    # REMEMBER TO CHECK THIS WHEN DOING MULTIPLE RUNS (although technically it shouldn't matter....)
    weight_df.drop(['observer', 'run'], axis=1, inplace=True)

    # Sum for each run (i.e. group by everything but time)
    time_summed_weights = weight_df.groupby(level=list(weight_df.index.names[:-1])).sum().unstack('var')

    if target is not None:
        target_weights = time_summed_weights.xs(target, level='target', axis=1)
    return target_weights.fillna(0.0)  # Nans map to no outliers


if __name__ == "__main__":
    observer = 'Bravo'
    target = 'Alfa'
    n_nodes = 6
    n_metrics = 9

    results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-03-16-45-26"

    with pd.get_store('/home/bolster/src/aietes/results/outlier_backup.h5') as store:
        target_weights_dict = {}
        for runkey in store.keys():
            print runkey
            target_weights_dict[runkey] = summed_outliers_per_weight(store.get(runkey), observer, n_metrics,
                                                                     target=target)
    target_weights = pd.concat(target_weights_dict, names=['run'] + target_weights_dict[runkey].index.names)
    # These results handily confirm that there is a 'signature' of badmouthing as RandomFlatWalk was incorrectly configured.
    #
    # Need to:
    # 1. Perform multi-run tolerance analysis of metrics (i.e. turn the below into a boxplot)
    # 2. Perform cross correlation analysis on metrics across runs/behaviours (what metrics are redundant)
    known_good_features_d = {}
    for basekey in target_weights.keys():  # Parallelisable
        print basekey
        # Single DataFrame of all features against one behaviour
        var_weights = target_weights.apply(lambda s: s / target_weights[basekey], axis=0).dropna()
        known_good_features_d[basekey] = pd.concat(
            [feature_extractor(s.reset_index(), var) for var, s in var_weights.iteritems()],
            keys=var_weights.keys(), names=['var', 'metric'])



