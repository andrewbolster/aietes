#!/usr/bin/env python
import wxversion

wxversion.ensureMinimal("2.8")

import wx
import os
import logging
import functools
import cProfile
import threading

logging.basicConfig(level=logging.DEBUG)

from Views import EphyraNotebook, GenericFrame

from bounos import  BounosModel
from bounos.Metrics import *

from aietes.Tools.Memoize import lru_cache as memoize


_ROOT = os.path.abspath(os.path.dirname(__file__))

time_change_condition = threading.Condition()
sim_updated_condition = threading.Condition()


def log_and_call():
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			# set name_override to func.__name__
			logging.info("Entered %s (%s)" % (func.__name__, kwargs))
			return func(*args, **kwargs)

		return wrapper

	return decorator


def check_model():
	def decorator(func):
		@functools.wraps(func)
		def wrapper(self, *args, **kwargs):
			assert self.model_is_ready(), "Model not properly instantiated"
			return func(self, *args, **kwargs)

		return wrapper

	return decorator


class EphyraController():
	def __init__(self):
		self.model = BounosModel()
		self.view = None
		self.metrics = []

	def load_data_file(self, file_path):
		self.model.import_datafile(file_path)

	def model_is_ready(self):
		return self.model.is_ready

	def set_model(self, model):
		self.model = model

	@check_model()
	def update_model(self, *args, **kw):
		#Update Base Data
		self.model.update_data_from_sim(*args, **kw)
		# Update secondary metrics
		for metric in self.metrics:
			metric.update(self.model)

	@check_model()
	def metric_views(self, i=None, *args, **kw):
		if i is None:
			return [MetricView(metric, *args, **kw) for metric in self.metrics]
		elif isinstance(i, int):
			return MetricView(self.metrics[i], *args, **kw)
		else:
			raise NotImplementedError("Metrics Views must be addressed by int's or nothing! (%s)" % str(i))

	def set_view(self, view):
		self.view = view

	def run_simulation(self, config):
		#TODO will raise ConfigError on, well, config errors
		pass

	@check_model()
	def get_raw_positions_log(self):
		return self.model.p

	@check_model()
	def get_vector_names(self, i=None):
		return self.model.names if i is None else self.model.names[:]

	def get_n_vectors(self):
		try:
			return self.model.n
		except AttributeError:
			return 0

	@check_model()
	def get_model_title(self):
		return self.model.title

	@check_model()
	def get_extent(self):
		return self.model.environment

	@check_model()
	def get_final_tmax(self):
		if self.model.is_simulating:
			return self.model.simulation.tmax
		else:
			return self.model.tmax

	@check_model()
	def get_3D_trail(self, node=None, time_start=None, length=None):
		"""
		Return the [X:][Y:][Z:] trail for a given node from time_start backwards to
		a given length

		If no time given, assume the full time range
		"""
		time_start = time_start if time_start is not None else self.get_final_tmax()
		time_end = max(0 if length is None else (time_start - length), 0)

		if node is None:
			return [self.model.p[:][dimension][time_start:time_end:-1] for dimension in 0, 1, 2]
		else:
			return [self.model.p[node][dimension][time_start:time_end:-1] for dimension in 0, 1, 2]

	@check_model()
	@memoize()
	def get_fleet_configuration(self, time):
		"""
		   Returns a dict of the fleet configuration:

		   :param time: time index to calculate at
		   :type time int

		   :returns dict {positions, headings, avg_pos, avg_head, stdev_pos, stdev_head}
		   """
		positions = self.model.position_slice(time)
		avg_pos = np.average(positions, axis=0)
		_distances_from_avg_pos = map(lambda v: np.linalg.norm(v - avg_pos), positions)
		stddev_pos = np.std(_distances_from_avg_pos)
		headings = self.model.heading_slice(time)
		avg_head = np.average(headings, axis=0)
		_distances_from_avg_head = map(lambda v: np.linalg.norm(v - avg_head), headings)
		stddev_head = np.std(_distances_from_avg_head)

		return dict({'positions':
						 {
						 'pernode': positions,
						 'avg': avg_pos,
						 'stddev': stddev_pos,
						 'delta_avg': _distances_from_avg_pos
						 },
					 'headings':
						 {
						 'pernode': headings,
						 'avg': avg_head,
						 'stddev': stddev_head,
						 'delta_avg': _distances_from_avg_head
						 }
		})

	def get_heading_mag_max_min(self):
		mags = self.model.heading_mag_range()
		return (max(mags), min(mags))

	def get_position_stddev_max_min(self):
		stddevs = self.model.distance_from_average_stddev_range()
		return (max(stddevs), min(stddevs))


def main():
	app = wx.PySimpleApp()
	controller = EphyraController()
	app.frame = EphyraNotebook(controller)
	app.frame.Show()
	app.MainLoop()


def debug():
	cProfile.run('main()')

if __name__ == '__main__':
	main()


