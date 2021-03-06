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

import operator
import pandas as pd
import sklearn.ensemble as ske
import numpy as np
from time import gmtime, strftime
import itertools
from joblib import Parallel, delayed
from itertools import ifilter
from aietes.Tools import var_rename_dict, metric_rename_dict, categorise_dataframe, map_levels
from bounos.Analyses import Trust
from bounos.ChartBuilders.weight_comparisons import norm_weight


def generate_weighted_trust_perspectives(trust_observations, feat_weights, fair_filter=True, par=True):
    weighted_trust_perspectives = {}
    for k, w in feat_weights.to_dict().items():
        if k[0] != 'Fair' and fair_filter:
            continue
        weighted_trust_perspectives[k] = Trust.generate_node_trust_perspective(
            trust_observations,
            metric_weights=pd.Series(w),
            par=par
        )
    for key, trust in weighted_trust_perspectives.items():
        map_levels(weighted_trust_perspectives[key], var_rename_dict)

    return weighted_trust_perspectives


def generate_trust_metric_weights(trust_metric_names, exclude=None, max_emphasis=5):
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


def generate_run_comparative_outlier_tp(tf, sigma=None, good='good', good_lvl='bev'):
    """
    Looking across behaviours of a given run-observer set, use the >1std area beyond the all-time mean
    to assess "outlierishness".

    This must be applied on a per-observation basis (i.e. single run, single observer)
    :param tf:
    :param sigma:
    :param good:
    :param good_lvl:
    :return:
    """
    if sigma is not None:
        raise NotImplementedError("!Have't implemented")

    _mean = tf.xs(good, level=good_lvl).mean()
    _std = tf.xs(good, level=good_lvl).std()
    llim = _mean - _std
    ulim = _mean + _std
    uppers = (tf[tf > ulim] - ulim).reset_index()
    lowers = (llim - tf[tf < llim]).reset_index()
    outliers = pd.concat((uppers, lowers), keys=('upper', 'lower'), names=['range', 'id']).reset_index().set_index(
        ['var', 'run', 'observer', 'range', 't']
    )
    outliers.drop('id', axis=1, inplace=True)
    return outliers


def generate_run_comparative_outlier_frame(good, trust_frame, w, par=False):
    """
    Generates the weighted trust perspective
    Then for each observation run (i.e. each simulated nodes perspective), generate the
     comparative outlier trust assessment *across known behaviours*


    :param good:
    :param trust_frame:
    :param w:
    :param par:
    :return:
    """
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
        l_outliers.append(generate_run_comparative_outlier_tp(tf, good=good, good_lvl='var'))
    outlier = pd.concat(l_outliers).sort_index().dropna(how='all').reset_index()
    for k in w.keys():
        outlier[k] = w[k]
    outlier.set_index(['var', 't'], inplace=True)
    return outlier


def perform_weight_factor_run_comparative_outlier_analysis_on_trust_frame(trust_frame, good, min_emphasis=0, max_emphasis=3,
                                                                          min_sum=1, max_sum=None, extra=None,
                                                                          verbose=False, par=False,
                                                                          take_outlier_backup=False):
    """
    For a given trust frame perform grey weight factor distribution analysis to
    generate a frame with metrics/metric weights as a multiindex with columns for each comparison between behaviours

    :param min_emphasis:
    :param max_emphasis:
    :param extra:
    :param verbose:
    :param par:
    :param trust_frame:
    :param good: the baseline behaviour to assess against
    :param: max_metrics:
    :param: max_emphasis:
    :return:
    """
    # Extract trust metric names from frame
    trust_metrics = list(trust_frame.keys())
    if verbose:
        print("Using {0} Trust Metrics".format(trust_metrics))

    if max_emphasis - min_emphasis < 2:
        raise RuntimeError("Setting Max Emphasis <2 is pointless; Runs all Zeros")

    if extra is None:
        extra = ""

    if max_sum is None:
        max_sum = np.inf

    combinations = itertools.ifilter(np.any,
                                     itertools.product(
                                         xrange(
                                             min_emphasis, max_emphasis
                                         ), repeat=len(trust_metrics)
                                     )
                                     )
    combinations = itertools.ifilter(lambda v: min_sum <= np.sum(v) <= max_sum,
                                     combinations)
    combinations = sorted(combinations, key=lambda l: sum(map(abs, l)))
    if verbose:
        print("Using {0} combinations:".format(len(combinations)))
        print(combinations)
    if par:
        outliers = _run_comparative_outlier_par_inner_single_thread(combinations, good, trust_frame, trust_metrics, verbose=True)
    else:
        outliers = _run_comparative_outlier_single_thread_inner_par(combinations, good, trust_frame, trust_metrics, verbose=True)

    sums = pd.concat(outliers).reset_index()
    if take_outlier_backup:
        sums.to_hdf('/home/bolster/src/aietes/results/outlier_backup.h5', "{0}{1}_{2}".format(good, extra, max_emphasis))
    return sums


def _run_comparative_outlier_single_thread_inner_par(combinations, good, trust_frame, trust_metrics, verbose):
    outliers = []
    for i, w in enumerate(combinations):
        _w = norm_weight(w, trust_metrics)
        if verbose: print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), i, _w)
        outlier = generate_run_comparative_outlier_frame(good, trust_frame, _w, par=True)
        outliers.append(outlier)
    return outliers


def _run_comparative_outlier_par_inner_single_thread(combinations, good, trust_frame, trust_metrics, verbose):
    outliers = Parallel(n_jobs=-1, verbose=int(verbose) * 50)(delayed(generate_run_comparative_outlier_frame)
                                                              (good, trust_frame, norm_weight(w, trust_metrics), False)
                                                              for w in combinations)
    return outliers


def mean_t_delta(result, target_col=0):
    """
    Mean Delta T
    :param result:
    :param target_col: int: column index of targeted node
    :return:
    """
    target = result.values[:,target_col]
    cohort = np.delete(result.values, target_col, axis=1)
    #This is geometrically equivalant to doing a time-series mean but is 3 times faster
    # (i.e. (np.mean(cohort, axis=1)-target).mean())

    # NOTE You changed nanmean to mean here for a performance boost of 20%
    return np.subtract(np.mean(cohort),target).mean()


def generate_mean_t_delta_frame(trust_frame, w, target, par=False):
    """

    :param trust_frame:
    :param w:
    :param target:
    :param par:
    :return:
    """

    # TODO experiment with wether it's better to parallelise the inner or outer version of this.... (i.e. par/w vs par/single
    weighted_trust_perspectives = Trust.generate_node_trust_perspective(trust_frame,
                                                                        flip_metrics=[],
                                                                        metric_weights=w,
                                                                        par=par)
    target_col = weighted_trust_perspectives.columns.tolist().index(target)

    l_outliers = []
    for i, tf in weighted_trust_perspectives.groupby(level=['var', 'run', 'observer']):
        tf=tf.drop(i[-1], axis=1) #Drop observer from MDT calc
        l_outliers.append((i, mean_t_delta(tf, target_col=target_col)))
    outlier = pd.Series(dict(l_outliers))
    outlier.index.names = ['var', 'run', 'observer']
    outlier = outlier.sort_index().dropna(how='all').reset_index()
    for k in w.keys():
        outlier[k] = w[k]
    outlier.set_index(['var'], inplace=True)
    return outlier


def perform_weight_factor_target_mean_t_delta_analysis_on_trust_frame(trust_frame, min_emphasis=0, max_emphasis=3,
                                                                      min_sum = 1, max_sum = None,
                                                                      extra=None, verbose=False, par=False,
                                                                      target='Alfa', excluded=[],
                                                                      take_outlier_backup=False):
    """
    For a given trust frame perform grey weight factor distribution analysis to
    generate a frame with metrics/metric weights as a multiindex with columns for each comparison between behaviours

    :param min_emphasis:
    :param max_emphasis:
    :param extra:
    :param verbose:
    :param par:
    :param target:
    :param excluded:
    :param trust_frame:
    :param: max_metrics:
    :param: max_emphasis:
    :param: target:
    :return:
    """
    # Extract trust metric names from frame
    trust_metrics = list(trust_frame.keys())
    if max_emphasis - min_emphasis < 2:
        raise RuntimeError("Setting Max Emphasis <2 is pointless; Runs all Zeros")

    if extra is None:
        extra = ""

    if max_sum is None:
        max_sum = np.inf

    combinations = itertools.ifilter(np.any,
                                     itertools.product(
                                         xrange(
                                             min_emphasis, max_emphasis
                                         ), repeat=len(trust_metrics)
                                     )
                                     )
    combinations = itertools.ifilter(lambda v: min_sum <= np.sum(v) <= max_sum,
                                     combinations)
    combinations = sorted(combinations, key=lambda l: sum(map(abs, l)))
    if verbose:
        print("Using {0} combinations:".format(len(combinations)))
        print(combinations)
    if par:
        outliers = _target_mean_t_delta_par_inner_single_thread(combinations, trust_frame, trust_metrics, target,
                                                                verbose=True)
    else:
        outliers = _target_mean_t_delta_single_thread_inner_par(combinations, trust_frame, trust_metrics, target,
                                                                verbose=True)

    sums = pd.concat(outliers).reset_index()
    if take_outlier_backup:
        sums.to_hdf('/home/bolster/src/aietes/results/outlier_backup.h5',
                    "{0}{1}_{2}".format("meandelta", extra, max_emphasis))
    return sums


def _target_mean_t_delta_single_thread_inner_par(combinations, trust_frame, trust_metrics, target, verbose):
    outliers = []
    for i, w in enumerate(combinations):
        if verbose: print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), i, w)
        outlier = generate_mean_t_delta_frame(trust_frame, norm_weight(w, trust_metrics), target, par=True)
        outliers.append(outlier)
    return outliers


def _target_mean_t_delta_par_inner_single_thread(combinations, trust_frame, trust_metrics, target, verbose):
    outliers = Parallel(n_jobs=-1, verbose=int(verbose) * 50)(delayed(generate_mean_t_delta_frame)
                                                              (trust_frame, norm_weight(w, trust_metrics), target,
                                                               False)
                                                              for w in combinations)
    return outliers


def summed_outliers_per_weight(weight_df, observer, n_metrics, target=None, signed=False):
    # Select Perspective here
    weight_df = weight_df[weight_df.observer == observer]
    weight_df = categorise_dataframe(weight_df)

    # Metrics are the last sector of the frame, set these as leading indices
    metric_keys = list(weight_df.keys()[-n_metrics:])
    weight_df.set_index(metric_keys + ['var', 't'], inplace=True)

    # REMEMBER TO CHECK THIS WHEN DOING MULTIPLE RUNS (although technically it shouldn't matter....)
    weight_df = weight_df.drop(['observer', 'run'], axis=1)

    # TODO: Assert abs(signed) = unsigned
    # Sum for each run (i.e. group by everything but time)
    if signed:
        d = {'upper': 1, 'lower': -1}
        r = weight_df['range'].apply(d.get)

        time_summed_outliers = \
            weight_df \
                .drop('range', axis=1) \
                .apply(lambda c: c * r) \
                .groupby(level=list(weight_df.index.names[:-1])) \
                .sum() \
                .unstack('var')
    else:
        time_summed_outliers = \
            weight_df \
                .groupby(level=list(weight_df.index.names[:-1])) \
                .sum() \
                .unstack('var')

    if target is not None:
        target_weights = time_summed_outliers.xs(target, level='target', axis=1)
    return target_weights.fillna(0.0)  # Nans map to no outliers


def feature_extractor(df, target, raw=False, n_estimators=128):
    data = df.drop(target, axis=1)
    reg = ske.ExtraTreesRegressor(n_jobs=-1, n_estimators=n_estimators)
    reg.fit(data, df[target])
    if raw:
        result = target, reg
    else:
        result = pd.Series(dict(zip(data.keys(), reg.feature_importances_)))
    return result


def target_weight_feature_extractor(target_weights, comparison=None, raw=False, n_estimators=128):
    if comparison is None:
        comparison = np.subtract

    known_good_features_d = {}
    for basekey in target_weights.keys():  # Parallelisable
        print basekey
        # Single DataFrame of all features against one behaviour
        var_weights = target_weights.apply(lambda s: comparison(s, target_weights[basekey]), axis=0).dropna()
        # Ending up with [basekey,var] as "Baseline" and "target" behaviours
        if raw:
            known_good_features_d[basekey] = [
                feature_extractor(s.reset_index(), var, raw=True, n_estimators=n_estimators) for var, s in
                var_weights.iteritems()]
        else:
            known_good_features_d[basekey] = \
                pd.concat([feature_extractor(s.reset_index(), var, n_estimators=n_estimators) for var, s in
                           var_weights.iteritems()],
                          keys=var_weights.keys(), names=['var', 'metric'])

    return known_good_features_d


def dataframe_weight_filter(df, keys):
    indexes = [(df.index.get_level_values(k) == 0.0) for k in keys]
    return df.loc[reduce(operator.and_, indexes)]


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


if __name__ == "__main__":
    observer = 'Bravo'
    target = 'Alfa'
    n_nodes = 6
    n_metrics = 9

    results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"

    with pd.get_store(results_path + "/outliers.bkup.h5") as store:
        target_weights_dict = {}
        weight_df = store.get(store.keys()[0])
        summed_outliers_per_weight(weight_df, observer, n_metrics, target=target)


def build_outlier_weights(h5_path, observer, target, n_metrics, signed=False):
    """Outliers should have keys of runs
    :param h5_path:
    """
    with pd.get_store(h5_path) as store:
        keys = store.keys()

    target_weights_dict = {}
    for runkey in filter(lambda s: s.startswith('/CombinedTrust'), keys):
        with pd.get_store(h5_path) as store:
            print runkey
            target_weights_dict[runkey] = summed_outliers_per_weight(store.get(runkey),
                                                                     observer, n_metrics,
                                                                     target=target,
                                                                     signed=False)

    if runkey:
        joined_target_weights = pd.concat(
            target_weights_dict, names=['run'] + target_weights_dict[runkey].index.names
        ).reset_index('run', drop=True).sort_index()
    else:
        raise ValueError("Doesn't look like there's any CombinedTrust records in that h5_path... {0}".format(h5_path))

    return joined_target_weights


def calc_correlations_from_weights(weights):
    def calc_correlations(base, comp, index=0):
        dp_r = (comp / base).reset_index()
        return dp_r.corr()[index][:-1]

    _corrs = {}
    for base, comp in itertools.permutations(weights.keys(), 2):
        _corrs[(base, comp)] = \
            calc_correlations(weights[base],
                              weights[comp])

    corrs = pd.DataFrame.from_dict(_corrs).T.rename(columns=metric_rename_dict)
    map_levels(corrs, var_rename_dict, 0)
    map_levels(corrs, var_rename_dict, 1)
    corrs.index.set_names(['Control', 'Misbehaviour'], inplace=True)
    return corrs


def drop_metrics_from_weights_by_key(target_weights, drop_keys):
    reset_by_keys = target_weights.reset_index(level=drop_keys)
    zero_indexes = (reset_by_keys[drop_keys] == 0.0).all(axis=1)
    dropped_target_weights = reset_by_keys[zero_indexes].drop(drop_keys, 1)
    return dropped_target_weights
