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

import numpy as np

from bounos import DataPackage, Analyses
from aietes.Tools import mkpickle


def find_convergence(data, *args, **kwargs):
    """
    Return the Time of estimated convergence of the fleet along with some certainty value
    using the Average of Inter Node Distances metric
    i.e. the stability of convergence
    :param args:
    :param kwargs:
    :param data:
    Args:
        data(DataPackage)
    Raises:
        AssertionError if data is not DataPackage
    Returns:
        tuple of:
            detection_points, metrics (both just data)

    """
    # TODO This isn't done at all
    assert isinstance(data, DataPackage)
    detection_points = data
    metrics = data
    return detection_points, metrics


def detect_misbehaviour(data, metric="PerNode_Internode_Distance_Avg",
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
                :param args:
                :param kwargs:
    """
    metric = Analyses.get_valid_metric(metric)

    metric.update(data)
    # IND has the highlight data to be the average of internode distances
    # TODO implement scrolling stddev calc to adjust smearing value (5)
    potential_misbehavers = {}
    detection_envelope = np.zeros(data.tmax, dtype=np.float64)
    deviance = np.zeros((data.tmax, data.n), dtype=np.float64)

    rolling_detections = [[]] * data.tmax
    confirmed_detections = [[]] * data.tmax
    confirmation_envelope = 10

    for t in range(data.tmax):
        try:
            deviance[t] = metric.data[t] - metric.highlight_data[t]
        except TypeError as e:
            raise TypeError("{0!s}:{1!s}".format(metric.__class__.__name__, e))

        # Select culprits that are deviating by 1 sigma/frac from the norm
        detection_envelope[t] = this_detection_envelope = np.std(
            deviance[t]) / stddev_frac
        culprits = [False]
        # None is both not True and not False
        if metric.signed is not False:
            # Positive Swing
            culprits = (
                metric.data[t] > (metric.highlight_data[t] + this_detection_envelope))
        elif metric.signed is not True:
            # Negative Swing
            culprits = (
                metric.data[t] < (metric.highlight_data[t] - this_detection_envelope))
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

            # Check if culprit is in all the last $envelope detection lists
            if all(culprit in detection_list for detection_list in rolling_detections[t - confirmation_envelope:t]) \
                    and t > confirmation_envelope:
                confirmed_detections[t].append(culprit)

    return {'detections': np.asarray(confirmed_detections),
            'detection_envelope': detection_envelope,
            'suspicions': potential_misbehavers,
            'deviance': deviance,
            'metrics': metric.data}


def deviation_from_metric(data, *args, **kwargs):
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
                :param args:
                :param kwargs:
    """

    metric_arg = kwargs.get("metric", "PerNode_Internode_Distance_Avg")
    metric = Analyses.get_valid_metric(metric_arg)
    metric.update(data)
    # IND has the highlight data to be the average of internode distances
    # TODO implement scrolling stddev calc to adjust smearing value (5)
    stddev = np.zeros(data.tmax, dtype=np.float64)
    deviance = np.zeros((data.tmax, data.n), dtype=np.float64)

    for t in range(data.tmax):
        try:
            deviance[t] = abs(metric.data[t] - metric.highlight_data[t])
        except TypeError as e:
            raise TypeError("{0!s}:{1!s}".format(metric.__class__.__name__, e))

        stddev[t] = np.std(deviance[t])

    return {'stddev': stddev,
            'deviance': deviance,
            'metrics': metric.data}


def deviance_assessor(data, metrics, suspects_only=False, stddev_frac=2, override = False, window=600,
                      tmax = None, **kwargs):
    # Combine multiple metrics detections into a general trust rating per node
    # over time.
    """

    :param data:
    :param metrics:
    :param suspects_only:
    :param args:
    :param kwargs:
    :return: :raise ValueError:
    """
    if not isinstance(metrics, list):
        raise ValueError("metrics should be passed a list of analyses")
    if not isinstance(data, DataPackage.DataPackage):
        raise ValueError("data should be a DataPackage, got {}".format(type(data)))
    tmax = data.tmax if tmax is None else tmax
    n_met = len(metrics)
    n_nodes = data.n
    deviance_accumulator = np.zeros((n_met, tmax, n_nodes), dtype=np.float64)

    deviance_accumulator.fill(0.0)
    for m, metric in enumerate(metrics):
        # Get Detections, Stddevs, Misbehavors, Deviance from
        # Detect_MisBehaviour
        if override:
            results = deviation_from_metric(data, metric=metric)
            print("No misbehavors given, assuming everything")
            misbehavors = {suspect: range(data.tmax)
                           for suspect in range(data.n)}
        else:
            results = detect_misbehaviour(data, metric=metric, stddev_frac=stddev_frac)
            misbehavors = results['suspicions']

        stddev, deviance = results['detection_envelope'], results['deviance']

        # Using culprits as a filter only searches for distrust, not 'good
        # behaviour'
        if suspects_only:
            for culprit, times in misbehavors.iteritems():
                deviance_accumulator[m, np.array(times), culprit] = (
                    np.abs(np.divide(deviance[np.array(times), culprit],
                                     stddev[np.array(times)].clip(min=np.finfo(np.float64).eps)))
                )
        else:
            for culprit in xrange(n_nodes):
                # for each target node, the deviance_acc value for time t is the
                # absolute value of the devience over the
                deviance_accumulator[m, :, culprit] = (
                    np.abs(
                        np.divide(deviance[:, culprit],
                                  stddev.clip(min=np.finfo(np.float64).eps))
                    )
                )

    windowed_deviance = np.zeros((tmax, n_nodes), dtype=np.float64)
    lag_lead_deviance = np.zeros((tmax, n_nodes), dtype=np.float64)

    # Prod-Sum smooth individual metrics based on a window
    for t in range(tmax):
        head = max(0, t - window)
        lag_lead_deviance[t] = np.sum(
            np.prod(deviance_accumulator[:, head:t, :], axis=0),
            axis=0)
        windowed_deviance[
            t] = lag_lead_deviance[t] - (t - head)

    return deviance_accumulator, windowed_deviance


def behaviour_identification(deviance, windowed_deviance, metrics, names=None, verbose=False):
    """
    #TODO THIS DOES NOT IDENTIFY BEHAVIOUR
    Attempts to detect and guess malicious/'broken' node
    Deviance is unitless, in a shape [metrics,t,nodes]
    :param deviance:
    :param windowed_deviance:
    :param metrics:
    :param names:
    :param verbose:
    """
    detection_sums = np.sum(
        deviance, axis=1) - deviance.shape[1]                   # Removes the 1.0 initial bias (i.e. N nodes)
    detection_totals = np.sum(detection_sums, axis=1)
    detection_subtot = np.argmax(detection_sums, axis=1)
    detection_max = np.argmax(np.sum(detection_sums, axis=0))
    trust_average = np.average(windowed_deviance, axis=0)       # Per Node
    trust_stdev = np.std(trust_average)
    prime_distrusted_node = np.argmax(trust_average)
    #mkpickle("trust", windowed_deviance)
    # In reality, the below is basically run on everyone
    if trust_stdev > 100: # This is arbitrary; should be dependant on N, T, etc.
        if verbose:
            print("Untrustworthy behaviour detected")
            if names is None:
                print("\n".join(["{0!s}:{1:d}({2:f})".format(metrics[i].label, m_subtot, detection_totals[
                    i]) for i, m_subtot in enumerate(detection_subtot)]))
                print("Prime Suspect:{0!s}:{1!s}".format(
                    prime_distrusted_node, str(trust_average[prime_distrusted_node])))
            else:
                print("\n".join(["{0!s}:{1!s}({2:f})".format(metrics[i].label, names[
                    m_subtot], detection_totals[i]) for i, m_subtot in enumerate(detection_subtot)]))
                print("Prime Suspect:{0!s}:{1!s}".format(
                    names[prime_distrusted_node], str(trust_average[prime_distrusted_node])))

    confidence = (trust_average[prime_distrusted_node] - np.average(trust_average)) / np.std(trust_average)

    result = {"suspect": prime_distrusted_node,
              "suspect_name": names[prime_distrusted_node] if names is not None else None,
              "suspect_distrust": trust_average[prime_distrusted_node],
              "suspect_confidence": confidence,
              "trust_stdev": trust_stdev,
              "trust_average": trust_average,
              "detection_totals": detection_totals}
    return result
