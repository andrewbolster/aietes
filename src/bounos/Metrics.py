__author__ = 'andrewbolster'

import logging
import numpy as np

from aietes.Tools import mag


class Metric(object):
    """ This Axes provides data plotting tools

    data format is array(t,n) where n can be 1 (i.e. linear array)

    The 'wanted' functionality acts as a boolean filter, i.e.
    w = [010]
    d = [[111][222][333]]
    p = d[:,w}

    Metrics should always return so that observations are addressable by time primarily
    i.e. result[0] is the metric state of the system at time zero
    """
    signed = None

    # Assumes that data is constant and only needs to be selected per node
    def __init__(self, *args, **kw):
        if not hasattr(self, 'label'):
            self.label = kw.get('label', self.__class__.__name__.replace("_", " "))
        self.highlight_data = kw.get('highlight_data', None)
        self.data = kw.get('data', None)
        self.signed = kw.get('signed', self.signed) # True is positive, False is Negative, None isn't
        self.ndim = 0
        if __debug__: logging.debug("%s" % self)

    def generator(self, data):
        return data

    def __repr__(self):
        return "Metric: %s with %s entries" % (
        self.label,
        self.data[0] if self.data is not None else None
        )

    def update(self, data = None):
        if data is None:
            self.data = np.asarray(self.generator(self.data))
        else:
            self.data = np.asarray(self.generator(data))
        if hasattr(self.data, 'ndim'):
            self.ndim = self.data.ndim
        else:
            self.ndim = 1


class StdDev_of_Distance(Metric):
    def generator(self, data):
        return data.distance_from_average_stddev_range()


class StdDev_of_Heading(Metric):
    def generator(self, data):
        return data.heading_stddev_range()


class Avg_Mag_of_Heading(Metric):
    def generator(self, data):
        return data.heading_mag_range()


class Deviation_Of_Heading(Metric):
    """
    Measured the per node deviation from the fleet path, therefore the 'average' is zero
    However, since INHD is unsigned, this isn're really the case, so take the avg of the vals
    """
    label = "INHD"
    signed = True
    def generator(self, data):

        vals = np.asarray([data.deviation_from_at(data.average_heading(time), time) for time in range(int(data.tmax))])
        self.highlight_data=np.average(vals, axis=1)
        return vals

class PerNode_Speed(Metric):
    label = "Node Speed"
    signed = False
    def generator(self, data):
        self.highlight_data = [mag(data.average_heading(time)) for time in range(int(data.tmax))]
        return [map(mag, data.heading_slice(time)) for time in range(int(data.tmax))]


class PerNode_Internode_Distance_Avg(Metric):
    label = "INDD"
    signed = True
    def generator(self, data):
        self.highlight_data = [data.inter_distance_average(time) for time in range(int(data.tmax))]
        return [data.distances_from_average_at(time) for time in range(int(data.tmax))]

