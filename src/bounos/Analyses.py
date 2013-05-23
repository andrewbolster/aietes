__author__ = 'andrewbolster'

import numpy as np

from DataPackage import DataPackage


def Find_Convergence(data, *args, **kwargs):
    """
    Return the Time of estimated convergence of the fleet along with some certainty value
    using the Average of Inter Node Distances metric
    i.e. the stability of convergence
    """
    assert isinstance(data, DataPackage)
    detection_points = data
    metrics = data
    return detection_points, metrics


def Detect_Misbehaviour(data, metric="PerNode_Internode_Distance_Avg", stddev_frac=1, *args, **kwargs):
    """
    Detect and identify if a node / multiple nodes are misbehaving.
    Currently misbehaviour is regarded as where the internode distance is significantly greater
        for any particular node of a significant period of time.
    Also can 'tighten' the detection bounds via fractions of \sigma
    Takes:
        metric:Metric("PerNode_Internode_Distance_Avg")
        stddev_frac:int(1)
    """
    import bounos.Metrics

    metric_arg = metric

    if not isinstance(metric_arg, str) and issubclass(metric_arg, bounos.Metrics.Metric):
        metric_class = metric_arg
    else:
        metric_name = metric_arg
        print(type(metric_name))
        try:
            metric_class = getattr(bounos.Metrics, metric_name)
        except Exception as e:
            print(metric_name)
            print(type(metric_name))
            raise e
    metric = metric_class()
    if metric is None:
        raise ValueError("No Metric! Cannot Contine")
    else:
        print("Performing Misbehaviour Detection with %s" % (metric.__class__.__name__))
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
            culprits = (metric.data[t] > (metric.highlight_data[t] + this_detection_envelope ))
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
            if all(culprit in detection_list for detection_list in
                   rolling_detections[t - confirmation_envelope:t]) and t > confirmation_envelope:
                confirmed_detections[t].append(culprit)

    return {'detections': np.asarray(confirmed_detections),
            'detection_envelope': detection_envelope,
            'suspicions': potential_misbehavers,
            'deviance': deviance,
            'metrics': metric.data
    }


def Deviation(data, *args, **kwargs):
    """
    Detect and identify if a node / multiple nodes are misbehaving.
    Currently misbehaviour is regarded as where the internode distance is significantly greater
        for any particular node of a significant period of time.

    """
    import bounos.Metrics

    metric_arg = kwargs.get("metric", "PerNode_Internode_Distance_Avg")
    if not isinstance(metric_arg, str) and issubclass(metric_arg, bounos.Metrics.Metric):
        metric_class = metric_arg
    else:
        metric_name = metric_arg
        print(type(metric_name))
        try:
            metric_class = getattr(bounos.Metrics, metric_name)
        except Exception as e:
            print(metric_name)
            print(type(metric_name))
            raise e
    metric = metric_class()
    if metric is None:
        raise ValueError("No Metric! Cannot Contine")
    else:
        print("Performing Deviation analysis with %s" % (metric.__class__.__name__))
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
            'metrics': metric.data
    }


def Combined_Detection_Rank(data, metrics, *args, **kwargs):
    # Combine multiple metrics detections into a general trust rating per node over time.
    if not isinstance(metrics, list):
        raise ValueError("Should be passed a list of analyses")
    tmax = kwargs.get("tmax", data.tmax)
    window = kwargs.get("window", 600)
    override_detection = kwargs.get("override", False)
    n_met = len(metrics)
    n_nodes = data.n
    deviance_accumulator = np.zeros((n_met, tmax, n_nodes), dtype=np.float64)

    deviance_accumulator.fill(1.0)
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

        for culprit, times in misbehavors.iteritems():
            deviance_accumulator[m, np.array(times), culprit] = (
                np.abs(deviance[np.array(times), culprit] / stddev[np.array(times)]))

    deviance_windowed_accumulator = np.zeros((tmax, n_nodes), dtype=np.float64)
    deviance_lag_lead_accumulator = np.zeros((tmax, n_nodes), dtype=np.float64)

    for t in range(tmax):
        head = max(0, t - window)
        deviance_lag_lead_accumulator[t] = np.sum(np.prod(deviance_accumulator[:, head:t, :], axis=0), axis=0)
        deviance_windowed_accumulator[t] = deviance_lag_lead_accumulator[t] - (t - head)

    detection_sums = np.sum(deviance_accumulator, axis=1) - tmax
    detection_subtot = np.argmax(detection_sums, axis=1)
    detection_tot = np.argmax(np.sum(detection_sums, axis=0))
    print(detection_subtot)
    print(detection_tot)
    return deviance_accumulator, deviance_windowed_accumulator






