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

    metric_name = kwargs.get("metric", "PerNode_Internode_Distance_Avg")
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
    stddevs = []

    rolling_detections = [None] * data.tmax
    confirmed_detections = [None] * data.tmax
    confirmation_envelope = 10

    for t in range(data.tmax):
        culprit=None
        try:
            deltas = metric.data[t] - metric.highlight_data[t]
        except TypeError as e:
            raise TypeError("%s:%s"%(metric.__class__.__name__, e))
        stddev = np.std(deltas)
        stddevs.append(stddev)
        # Positive Swing
        if (metric.data[t] > metric.highlight_data[t] + stddev).any() and metric.signed is not False:
            culprit = metric.data[t].argmax()
            try:
                potential_misbehavers[culprit].append(t)
            except KeyError:
                potential_misbehavers[culprit] = [t]
            finally:
                rolling_detections[t]= culprit
            #print("%d:+%s"%(t,data.names[culprit]))
        # Negative Swing
        if (metric.data[t] < metric.highlight_data[t] - stddev).any() and metric.signed is not True:
            culprit = metric.data[t].argmin()
            try:
                potential_misbehavers[culprit].append(t)
            except KeyError:
                potential_misbehavers[culprit] = [t]
            finally:
                rolling_detections[t]= culprit
            #print("%d:-%s"%(t,data.names[culprit]))

        # $envelope detections in a row triggers confirmation

        if culprit is not None and all( d == culprit for d in rolling_detections[t-confirmation_envelope:t]):
            confirmed_detections[t]= culprit

    return np.asarray(confirmed_detections), np.asarray(stddevs), potential_misbehavers


