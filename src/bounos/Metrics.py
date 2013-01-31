__author__ = 'andrewbolster'

import logging
import numpy as np

from aietes.Tools import mag


class Metric(object):
	"""	This Axes provides data plotting tools

	data format is array(t,n) where n can be 1 (i.e. linear array)

	The 'wanted' functionality acts as a boolean filter, i.e.
	w = [010]
	d = [[111][222][333]]
	p = d[:,w}
	"""

	# Assumes that data is constant and only needs to be selected per node
	def __init__(self, *args, **kw):
		self.label = kw.get('label', self.__class__.__name__)
		self.highlight_data = kw.get('highlight_data', None)
		self.data = kw.get('data', np.zeros((0, 0)))
		self.ndim = 0
		if __debug__: logging.debug("%s" % self)

	def generator(self, data):
		return data

	def __repr__(self):
		return "PerNodeGraph_Axes: %s with %s values arranged as %s" % (
		self.label,
		len(self.data),
		self.data.shape
		)

	def update(self, data):
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
		return  data.heading_mag_range()


class Avg_of_InterNode_Distances(Metric): #REDUNDANT
	def generator(self, data):
		return [data.inter_distance_average(time) for time in range(int(data.tmax))]


class PerNode_Speed(Metric):
	def generator(self, data):
		self.highlight_data = [mag(data.average_heading(time)) for time in range(int(data.tmax))]
		return [map(mag, data.heading_slice(time)) for time in range(int(data.tmax))]


class PerNode_Internode_Distance_Avg(Metric):
	def generator(self, data):
		self.highlight_data = [data.inter_distance_average(time) for time in range(int(data.tmax))]
		return [data.distances_from_average_at(time) for time in range(int(data.tmax))]

