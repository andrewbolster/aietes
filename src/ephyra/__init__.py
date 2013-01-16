#!/usr/bin/env python
import wxversion
wxversion.ensureMinimal("2.8")

import wx
import os
import logging
import functools
import cProfile
import threading

logging.basicConfig(level = logging.DEBUG)

from bounos import DataPackage
from bounos.Metrics import *

from Views import Notebook as EphyraView

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


class EphyraController():
	def __init__(self):
		self.model = None
		self.view = None
		self.metrics = []

	def load_data_file(self, file_path):
		self.model.update_data_from_file(file_path)


	def set_model(self, model):
		self.model = model

	def update_model(self, *args, **kw):
		#Update Base Data
		self.model.update_data_from_sim(*args, **kw)
		# Update secondary metrics
		for metric in self.metrics:
			metric.update(self.model.data)

	def metric_views(self, i = None, *args, **kw):
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
		sim = AIETESThread(config)
		sim.run(callback = None)

	def get_raw_positions_log(self):
		return self.model.data.p

	def get_vector_names(self, i = None):
		return self.model.data.names if i is None else self.model.data.names[:]

	def get_extent(self):
		return self.model.data.environment()

	def get_final_tmax(self):
		if self.model.is_simulating():
			return self.model.simulation.tmax
		else:
			return self.model.data.tmax


def main():
	app = wx.PySimpleApp()
	controller = EphyraController()
	app.frame = EphyraView(controller, parent=None)

	app.frame.Show()
	app.MainLoop()


def debug():
	cProfile.run('main()')

if __name__ == '__main__':
	main()


