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


def Detect_Misbehaviour(data, *args, **kwargs):
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
    metric=metric_class()
    if metric is None:
        raise ValueError("No Metric! Cannot Contine")
    else:
        print("Performing %s with %s"%(__name__, metric.__class__.__name__))
    metric.update(data)
    # IND has the highlight data to be the average of internode distances
    #TODO implement scrolling stddev calc to adjust smearing value (5)
    potential_misbehavers = {}
    stddev = np.zeros((data.tmax), dtype=np.float64)
    deviance = np.zeros((data.tmax,data.n), dtype=np.float64)

    rolling_detections = [[]] * data.tmax
    confirmed_detections = [[]] * data.tmax
    confirmation_envelope = 10


    for t in range(data.tmax):
        try:
            deviance[t] = metric.data[t] - metric.highlight_data[t]
        except TypeError as e:
            raise TypeError("%s:%s"%(metric.__class__.__name__, e))

        stddev[t] = np.std(deviance[t])
        culprits=[False]
        if metric.signed is not False:
            # Positive Swing
            culprits = (metric.data[t] > (metric.highlight_data[t] + stddev[t]))
        elif metric.signed is not True:
            # Negative Swing
            culprits = (metric.data[t] < (metric.highlight_data[t] - stddev[t]))
        else:
            culprits = (metric.data[t] > (metric.highlight_data[t] + stddev[t])) or (metric.data[t] < (metric.highlight_data[t] - stddev[t]))

        for culprit in np.where(culprits)[0]:
            try:
                potential_misbehavers[culprit].append(t)
            except KeyError:
                potential_misbehavers[culprit] = [t]
            finally:
                rolling_detections[t].append(culprit)

            #Check if culprit is in all the last $envelope detection lists
            if all( culprit in detection_list for detection_list in rolling_detections[t-confirmation_envelope:t]) and t>confirmation_envelope:
                    confirmed_detections[t].append(culprit)

    return np.asarray(confirmed_detections), stddev, potential_misbehavers, deviance


def Combined_Detection_Rank(data, metrics, *args, **kwargs):
    # Combine multiple metrics detections into a general trust rating per node over time.
    if not isinstance(metrics,list):
        raise ValueError("Should be passed a list of analyses")
    tmax = kwargs.get("tmax", data.tmax)
    n_met = len(metrics)
    n_nodes = data.n
    deviance_accumulator = np.zeros((n_met, tmax, n_nodes), dtype=np.float64)

    deviance_accumulator.fill(1.0)
    for m,metric in enumerate(metrics):
        # Get Detections, Stddevs, Misbehavors, Deviance from Detect_MisBehaviour
        _,stddev,misbehavors,deviance = Detect_Misbehaviour(data, metric=metric)

        for culprit ,times in misbehavors.iteritems():
            deviance_accumulator[m,np.array(times), culprit] = (np.abs(deviance[np.array(times), culprit] / stddev[np.array(times)]))

    deviance_windowed_accumulator = np.zeros((tmax, n_nodes), dtype=np.float64)
    for t in range(tmax):
        deviance_windowed_accumulator[t]=np.sum(np.prod(deviance_accumulator[:,t-50:t,:],axis = 0), axis = 0)

    detection_sums = np.argmax(deviance_accumulator, axis=2)
    print(detection_sums)
    return deviance_accumulator, deviance_windowed_accumulator






