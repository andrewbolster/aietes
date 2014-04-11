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

from aietes.Tools import mag


class Metric(object):
    """
    This superclass provides abstracted methods for generating, updating and presenting
        simulation data

    Output data format is array(t,n) where n can be 1 (i.e. linear array)

    The 'wanted' functionality acts as a boolean filter, i.e.
    w = [010]
    d = [[111][222][333]]
    p = d[:,w}

    Metrics should always return so that observations are addressable by time primarily
    i.e. result[0] is the metric state of the system at time zero
    """
    signed = None
    drift_enabled = False

    # Assumes that data is constant and only needs to be selected per node
    def __init__(self, *args, **kw):
        """
        Args:
            label(str):metric name (optional)
            data(ndarray[t,n]): multi-node metric data (optional)
            highlight_data(ndarray[t]): data to be higlighted as the 'average' behaviour (optional)
            signed(bool): decide if the metric values are signed or not, i.e. if deviation is
                directional (lower is ok but higher is bad)
        """
        if not hasattr(self, 'label'):
            self.label = kw.get('label', self.__class__.__name__.replace("_", " "))
        self.highlight_data = kw.get('highlight_data', None)
        self.data = kw.get('data', None)
        self.signed = kw.get('signed', self.signed) # True is positive, False is Negative, None isn't
        self.ndim = 0
        self.n = None

    def __call__(self, *args, **kwargs):
        self.update(*args,**kwargs)
        return self

    def generator(self, data):
        raise NotImplementedError("Uninitialised Metric generator")

    def __repr__(self):
        return "Metric: %s with %s entries" % (
            self.label,
            self.n if self.n is not None else "Unknown"
        )

    def update(self, data=None):
        """
        Re Run the generator function across either the existing data or data passed in.
        Args:
            data (DataPackage): (optional)
        """
        if data is None:
            self.data = np.asarray(self.generator(self.data))
        else:
            self.data = np.asarray(self.generator(data))
        self.n = self.data[0]
        if hasattr(self.data, 'ndim'):
            self.ndim = self.data.ndim
        else:
            self.ndim = 1


class StdDev_of_Distance(Metric):
    """
    Measures the level of variation (stddev) of the distance between each node and the centre of
        the fleet
    """
    def generator(self, data):
        return data.distance_from_average_stddev_range()


class StdDev_of_Heading(Metric):
    """
    Measures the level of variation (stddev) of the mag of the heading distance from fleet
        average heading
    """
    def generator(self, data):
        return data.heading_stddev_range()


class Avg_Mag_of_Heading(Metric):
    """
    Measures the average node speed in the fleet across time
    """
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
        self.highlight_data = np.average(vals, axis=1)
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

class Drift_Error(Metric):
    label = "Drift(m)"
    signed = False
    drift_enabled = True

    def generator(self, data):
        self.highlight_data = data.drift_RMS()
        return data.drift_error().swapaxes(0,1)

class ECEA_Error(Metric):
    label = "ECEA Var (m)"
    signed = False
    ecea_enabled = True

    def generator(self, data):
        self.highlight_data = data.drift_RMS(source="intent")
        return data.drift_error(source="intent").swapaxes(0,1)

