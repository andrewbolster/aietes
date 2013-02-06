__author__ = 'andrewbolster'

from DataPackage import DataPackage
import numpy as np
from pykalman import KalmanFilter



def Find_Convergence(data, *args, **kwargs):
	"""
	Return the Time of estimated convergence of the fleet along with some certainty value
	using the Average of Inter Node Distances metric
	i.e. the stability of convergence
	"""
	assert isinstance(data, DataPackage)

def Detect_Misbehaviour(data, *args, **kwargs):
    """
    Detect and identify if a node / multiple nodes are misbehaving.

    Currently misbehaviour is regarded as where the internode distance is significantly greater
        for any particular node of a significant period of time.

    """
    from bounos.Metrics import PerNode_Internode_Distance_Avg as IND

    ind = IND()
    ind.update(data)
    # IND has the highlight data to be the average of internode distances
    #TODO implement scrolling stddev calc to adjust smearing value (5)
    potential_misbehavers = {}
    stddevs = []
    for t in range(data.tmax):
        deltas = ind.data[t] - ind.highlight_data[t]
        stddev = np.std(deltas)
        stddevs.append(stddev)
        if (ind.data[t] > ind.highlight_data[t]+stddev).any():
            culprit = ind.data[t].argmax()
            try:
                potential_misbehavers[culprit].append(t)
            except KeyError:
                potential_misbehavers[culprit]=[t]

    return (potential_misbehavers,stddevs)


