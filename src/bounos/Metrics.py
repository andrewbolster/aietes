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
            self.label = kw.get(
                'label', self.__class__.__name__.replace("_", " "))
        self.highlight_data = kw.get('highlight_data', None)
        self.data = kw.get('data', None)
        # True is positive, False is Negative, None isn't
        self.signed = kw.get('signed', self.signed)
        self.ndim = 0
        self.scale = None
        self.n = None

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
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


class StddevOfDistance(Metric):
    """
    Measures the level of variation (stddev) of the distance between each node and the centre of
        the fleet
    """
    scale = 'm'

    def generator(self, data):
        return data.distance_from_average_stddev_range()


class StddevOfHeading(Metric):
    """
    Measures the level of variation (stddev) of the mag of the heading distance from fleet
        average heading
    """

    def generator(self, data):
        return data.heading_stddev_range()


class AvgMagOfHeading(Metric):
    """
    Measures the average node speed in the fleet across time
    """

    def generator(self, data):
        return data.heading_mag_range()


class DeviationOfHeading(Metric):
    """
    Measured the per node deviation from the fleet path, therefore the 'average' is zero
    However, since INHD is unsigned, this isn're really the case, so take the avg of the vals
    """
    label = "INHD($ms^-1)$"
    signed = True

    def generator(self, data):
        vals = np.asarray([data.deviation_from_at(
            data.average_heading(time), time) for time in range(int(data.tmax))])
        self.highlight_data = np.average(vals, axis=1)
        return vals


class PernodeSpeed(Metric):
    label = "Node Speed ($ms^-1$)"
    signed = False

    def generator(self, data):
        self.highlight_data = [
            mag(data.average_heading(time)) for time in range(int(data.tmax))]
        return [map(mag, data.heading_slice(time)) for time in range(int(data.tmax))]


class PernodeInternodeDistanceAvg(Metric):
    label = "INDD ($m$)"
    signed = True

    def generator(self, data):
        self.highlight_data = [
            data.inter_distance_average(time) for time in range(int(data.tmax))]
        return [data.distances_from_average_at(time) for time in range(int(data.tmax))]


class DriftError(Metric):
    label = "Drift($m$)"
    signed = False
    drift_enabled = True

    def generator(self, data):
        """

        :param data:
        :return:
        """
        self.highlight_data = data.drift_rms()
        return data.drift_error().swapaxes(0, 1)


class EceaError(Metric):
    label = "Corrected Drift ($m$)"
    signed = False
    ecea_enabled = True

    def generator(self, data):
        """

        :param data:
        :return:
        """
        self.highlight_data = data.drift_rms(source="intent")
        return data.drift_error(source="intent").swapaxes(0, 1)

#
# class Packet_Loss_Rate(Metric):
# label = "Packet Loss Rate (%p\%%)"
# signed = False
#
#     def generator(self, data):
#         return data.plr()
