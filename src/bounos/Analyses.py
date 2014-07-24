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
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

__author__ = 'andrewbolster'

import numpy as np
from pandas.stats.moments import ewma

from DataPackage import DataPackage
from aietes.Tools import mkpickle

def Find_Convergence(data, *args, **kwargs):
    """
    Return the Time of estimated convergence of the fleet along with some certainty value
    using the Average of Inter Node Distances metric
    i.e. the stability of convergence
    Args:
        data(DataPackage)
    Raises:
        AssertionError if data is not DataPackage
    Returns:
        tuple of:
            detection_points, metrics (both just data)

    """
    #TODO This isn't done at all
    assert isinstance(data, DataPackage)
    detection_points = data
    metrics = data
    return detection_points, metrics


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
    if isinstance(metric_arg, type) and hasattr(metric_arg,'mro') and metric_arg.mro()[-2].__name__ == 'Metric':
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


def Detect_Misbehaviour(data, metric="PerNode_Internode_Distance_Avg",
                        stddev_frac=1, *args, **kwargs):
    """
    Detect and identify if a node / multiple nodes are misbehaving.
    Currently misbehaviour is regarded as where the internode distance is significantly greater
        for any particular node of a significant period of time.
    Also can 'tighten' the detection bounds via fractions of \sigma
    Args:
        data(DataPackage)
        metric(Metric): What metric to use for detection
            (optional:"PerNode_Internode_Distance_Avg")
        stddev_frac(int): Adjust the detection threshold (optional:1)
    Raises:
        ValueError if metric cannot be found as a module (from bounos.Metrics)
        TypeError if metric deviation calculation bugs out
    Returns:
        Dict of:
            'detections': ndarray(t,list of names)
                misbehaviours 'confirmed' over the envelope period,
            'detection_envelope': ndarray(t,dtype=float)
                the detection range (stddev(deviance)/frac)
            'suspicions': [names][list of ints]
                list of times each node was under suspicion
            'deviance': ndarray(t,n, dtype=float)
                raw deviance from the mean for each node for each t
            'metrics': as metric.data
                Raw metric data
    """
    metric = get_valid_metric(metric)

    metric.update(data)
    # IND has the highlight data to be the average of internode distances
    #TODO implement scrolling stddev calc to adjust smearing value (5)
    potential_misbehavers = {}
    detection_envelope = np.zeros((data.tmax), dtype=np.float64)
    deviance = np.zeros((data.tmax, data.n), dtype=np.float64)

    rolling_detections = [[]] * data.tmax
    confirmed_detections = [[]] * data.tmax
    confirmation_envelope = 10

    for t in range(data.tmax):
        try:
            deviance[t] = metric.data[t] - metric.highlight_data[t]
        except TypeError as e:
            raise TypeError("%s:%s" % (metric.__class__.__name__, e))

        # Select culprits that are deviating by 1 sigma/frac from the norm
        detection_envelope[t] = this_detection_envelope = np.std(deviance[t]) / stddev_frac
        culprits = [False]
        # None is both not True and not False
        if metric.signed is not False:
            # Positive Swing
            culprits = (metric.data[t] > (metric.highlight_data[t] + this_detection_envelope))
        elif metric.signed is not True:
            # Negative Swing
            culprits = (metric.data[t] < (metric.highlight_data[t] - this_detection_envelope))
        else:
            culprits = (metric.data[t] > (metric.highlight_data[t] + this_detection_envelope)) or (
                metric.data[t] < (metric.highlight_data[t] - this_detection_envelope))

        for culprit in np.where(culprits)[0]:
            try:
                potential_misbehavers[culprit].append(t)
            except KeyError:
                potential_misbehavers[culprit] = [t]
            finally:
                rolling_detections[t].append(culprit)

            #Check if culprit is in all the last $envelope detection lists
            if all(culprit in detection_list for detection_list in rolling_detections[t - confirmation_envelope:t]) \
                    and t > confirmation_envelope:
                confirmed_detections[t].append(culprit)

    return {'detections': np.asarray(confirmed_detections),
            'detection_envelope': detection_envelope,
            'suspicions': potential_misbehavers,
            'deviance': deviance,
            'metrics': metric.data}


def Deviation(data, *args, **kwargs):
    """
    Calculate simple, absolute, deviance across the metric set.

    Args:
        data(DataPackage)
        metric(str/Metric):Metric (optional:"PerNode_Internode_Distance_Avg")
        stddev_frac(int): Adjust the detection threshold (optional:1)
    Raises:
        ValueError if metric cannot be found as a module (from bounos.Metrics)
        TypeError if metric deviation calculation bugs out
    Returns:
        dict of:
            'stddev':ndarray(t, dtype=float)
                stddev of deviance per time slot
            'deviance': ndarray(t,n, dtype=float)
                Absolute deviance from metric highlight data
            'metrics': as metric.data
                Raw metric data
    """
    import bounos.Metrics

    metric_arg = kwargs.get("metric", "PerNode_Internode_Distance_Avg")
    metric = get_valid_metric(metric_arg)
    metric.update(data)
    # IND has the highlight data to be the average of internode distances
    #TODO implement scrolling stddev calc to adjust smearing value (5)
    stddev = np.zeros((data.tmax), dtype=np.float64)
    deviance = np.zeros((data.tmax, data.n), dtype=np.float64)

    for t in range(data.tmax):
        try:
            deviance[t] = abs(metric.data[t] - metric.highlight_data[t])
        except TypeError as e:
            raise TypeError("%s:%s" % (metric.__class__.__name__, e))

        stddev[t] = np.std(deviance[t])

    return {'stddev': stddev,
            'deviance': deviance,
            'metrics': metric.data}


def Combined_Detection_Rank(data, metrics, suspects_only=False, *args, **kwargs):
    # Combine multiple metrics detections into a general trust rating per node over time.
    if not isinstance(metrics, list):
        raise ValueError("Should be passed a list of analyses")
    tmax = kwargs.get("tmax", data.tmax)
    window = kwargs.get("window", 600)
    override_detection = kwargs.get("override", False)
    n_met = len(metrics)
    n_nodes = data.n
    deviance_accumulator = np.zeros((n_met, tmax, n_nodes), dtype=np.float64)

    deviance_accumulator.fill(0.0)
    for m, metric in enumerate(metrics):
        # Get Detections, Stddevs, Misbehavors, Deviance from Detect_MisBehaviour
        if override_detection:
            results = Deviation(data, metric=metric)
            print("No misbehavors given, assuming everything")
            misbehavors = {suspect: range(data.tmax) for suspect in range(data.n)}
        else:
            results = Detect_Misbehaviour(data, metric=metric)
            misbehavors = results['suspicions']

        stddev, deviance = results['detection_envelope'], results['deviance']

        # Using culprits as a filter only searches for distrust, not 'good behaviour'
        if suspects_only:
            for culprit, times in misbehavors.iteritems():
                deviance_accumulator[m, np.array(times), culprit] = (
                    np.abs(np.divide(deviance[np.array(times), culprit], stddev[np.array(times)].clip(min=np.finfo(np.float64).eps)))
                )
        else:
            for culprit in xrange(n_nodes):
                deviance_accumulator[m, : , culprit] = (
                    np.abs(np.divide(deviance[:, culprit], stddev.clip(min=np.finfo(np.float64).eps)))
                )


    windowed_trust_accumulator = np.zeros((tmax, n_nodes), dtype=np.float64)
    deviance_lag_lead_accumulator = np.zeros((tmax, n_nodes), dtype=np.float64)

    # Prod-Sum smooth individual metrics based on a window
    for t in range(tmax):
        head = max(0, t - window)
        deviance_lag_lead_accumulator[t] = np.sum(
            np.prod(deviance_accumulator[:, head:t, :], axis=0),
            axis=0)
        windowed_trust_accumulator[t] = deviance_lag_lead_accumulator[t] - (t - head)

    return deviance_accumulator, windowed_trust_accumulator


def behaviour_identification(deviance, trust, metrics, names=None, verbose=False):
    """
    Attempts to detect and guess malicious/'broken' behaviour
    Deviance is unitless, in a shape [metrics,t,nodes]
    """
    detection_sums = np.sum(deviance, axis=1) - deviance.shape[1] #Removes the 1.0 initial bias
    detection_totals = np.sum(detection_sums, axis=1)
    detection_subtot = np.argmax(detection_sums, axis=1)
    detection_max = np.argmax(np.sum(detection_sums, axis=0))
    trust_average = np.average(trust, axis=0)
    trust_stdev = np.std(trust_average)
    prime_distrusted_node = np.argmax(trust_average)
    mkpickle("trust", trust)
    if trust_stdev > 100:
        if verbose:
            print("Untrustworthy behaviour detected")
            if names is None:
                print("\n".join(["%s:%d(%f)"%(metrics[i].label, m_subtot, detection_totals[i] ) for i,m_subtot in enumerate(detection_subtot)]))
                print("Prime Suspect:%s:%s"%(prime_distrusted_node, str(trust_average[prime_distrusted_node])))
            else:
                print("\n".join(["%s:%s(%f)"%(metrics[i].label, names[m_subtot], detection_totals[i]) for i,m_subtot in enumerate(detection_subtot)]))
                print("Prime Suspect:%s:%s"%(names[prime_distrusted_node], str(trust_average[prime_distrusted_node])))
    result = {"suspect": prime_distrusted_node,
              "suspect_name": names[prime_distrusted_node] if names is not None else None,
              "suspect_distrust": trust_average[prime_distrusted_node],
              "suspect_confidence": (trust_average[prime_distrusted_node]-np.average(trust_average))/np.std(trust_average),
              "trust_stdev": trust_stdev,
              "trust_average": trust_average,
              "detection_totals": detection_totals}
    return result

def GRC_t(measurements, bias='zero', rho=0.5, w=None):
    """
    Grey relational coefficient calculation per time slice

    measurement should be in [node,metric] order

    Assumes that measurements are zero-based (i.e. deviation from norm).

    Output is in the order [node,metric,[grc]]

    :param measurements:
    :return:
    """

    # Quick Checks
    n_nodes, n_metrics = measurements.shape
    zero_based = bias == 'zero'

    # Take the best reference sequence

    if zero_based:
        g = np.zeros(n_metrics)
    else:
        g = np.min(measurements, axis=0)

    # And the worst
    b = np.max(measurements, axis=0)

    def grc(delta):
        """
        GRC Inner function


        :param delta:
        :param rho:
        :return:
        """
        delta = abs(delta)
        upper = np.min(delta,axis=0) + (rho * np.max(delta, axis=0))
        lower = (delta) + (rho * np.max(delta, axis=0))
        with np.errstate(invalid='ignore', ):
            parterval = np.divide(upper,lower)
        return parterval

    intervals = np.asarray(map(grc, [measurements-g,measurements-b]))

    # Now we have the per-metric theta and sigmas
    # return as [node,metric,[interval]]
    return np.rollaxis(intervals,0,3)

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
    return np.average(c_intervals, axis=1,weights=None)

def T_kt(interval):
    """
    Generate a single trust value from a GRG
    1/ (1+ sigma^2/theta^2)
    :param interval:
    :return:
    """
    theta, sigma = interval
    return 1.0/(
        1.0+(
            np.power(sigma,2)/np.power(theta,2)
            )
    )

def dev_to_trust(per_metric_deviations):
    #rotate pmdev to node-primary ([node,metric])
    per_node_deviations = np.rollaxis(per_metric_deviations,0,3)
    grcs = map(GRC_t, per_node_deviations)
    grgs = map(GRG_t, grcs)
    trust_values = np.asarray([
        np.asarray([
            T_kt(interval) for interval in node
        ]) for node in grgs
    ])
    return trust_values