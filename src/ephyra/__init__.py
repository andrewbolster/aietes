#!/usr/bin/env python

import wx
import os
import logging
import functools
import cProfile
import threading
from pubsub import pub

logging.basicConfig(level = logging.DEBUG)

from bounos import DataPackage
from bounos.Metrics import *

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


class AIETESThread(threading.Thread):
	def __init__(self, config = None):
		"""
		@param simulation: the Aieted Simulation instance to control
		"""
		from aietes import Simulation

		threading.Thread.__init__(self)
		self._simulation = Simulation()
		self._info = self._simulation.prepare(waits = True, config = config)

		self.gui_time = 0
		self.sim_time = 0
		self.sim_time_max = self._info['sim_time']

		pub.subscribe(self.update_gui_time, 'update_gui_time')

	def update_gui_time(self, t):
		if __debug__: logging.info("RX time = %d" % t)
		self.gui_time = t

	def run(self, callback = None):
		"""
		Launch the simulation
		"""
		callback = self.callback if callback is None
		self._simulation.simulate(callback = self.callback)

	def callback(self):
		time_change_condition.acquire()
		while self.gui_time < self.sim_time and self.sim_time - self.gui_time > 10:
			"""
			When the simulation has caught up with the GUI; dump the data and yield to the gui again
			"""
			self.sim_time = self._simulation.now() - 1

			(p, v, names, environment) = self._simulation.currentState()

			if len(p[0]) == 0:
				# No Data Loaded
				break
			else:
				pub.sendMessage('update_data', p = p, v = v, names = names, environment = environment,
				                now = self._simulation.now())
				time_change_condition.wait()

		else:
			self.sim_time = self._simulation.now() - 1

		self._simulation.waiting = False
		time_change_condition.release()
		sim_updated_condition.acquire()
		sim_updated_condition.notify()
		sim_updated_condition.release()
		if __debug__: logging.info(
			"No Data to Load, yielding to simulation: T=%d, Ts=%d" % (self.gui_time, self.sim_time))

		return # Continue Simulating


class EphyraPanel(wx.Frame):
	def load_data(self, data_file):
		""" Load an external data file for plotting and navigation

		:param data_file: Aietes generated data for processing
		:type data_file: str

		Configures Plot area and adjusts control area
		"""
		try:
			self.data = DataPackage(data_file)
		except IOError as e:
			raise e

		self.reload_data()


	def reload_data(self):
		self.time_slider.SetRange(0, self.tmax)
		self.d_t = int((self.tmax + 1) / 100)

		self.resize_panel()


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

	controller = EphyraController(None)
	view = EphyraView(None)

	app.frame.Show()
	app.MainLoop()


def debug():
	cProfile.run('main()')

if __name__ == '__main__':
	main()


