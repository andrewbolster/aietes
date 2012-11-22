#!/usr/bin/env python

import wx
import os
import logging
import argparse
from mpl_toolkits.mplot3d.art3d import Line3DCollection

logging.basicConfig(level = logging.DEBUG)

import matplotlib

matplotlib.use('WXAgg')

from bounos import DataPackage

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm

matplotlib.rcParams.update({'font.size': 8})

import numpy as np


_ROOT = os.path.abspath(os.path.dirname(__file__))
WIDTH,HEIGHT = 8,5
SIDEBAR_WIDTH = 2

class EphyraFrame(wx.Frame):
	def __init__(self, *args, **kw):
		super(EphyraFrame, self).__init__(*args, **kw)
		self.log = logging.getLogger(self.__module__)
		parser = argparse.ArgumentParser(description = "GUI simulation and Analysis Suite for the Aietes framework")

		parser.add_argument('-o', '--open',
							dest = 'data_file', action = 'store', default = None,
							metavar = 'XXX.npz',
							help = 'Aietes DataPackage to be analysed'
		)
		parser.add_argument('-a', '--autostart',
			dest = 'autostart', action = 'store_true', default = False,
			help = 'Automatically launch animation on loading'
		)
		parser.add_argument('-l', '--loop',
			dest = 'loop', action = 'store_true', default = False,
			help = 'Loop animation'
		)

		self.args = parser.parse_args()

		self.paused = True
		self.telling_off = False # Used for Smartass Control
		self.t = 0
		self.d_t = 1

		self.CreateMenuBar()
		panel = wx.Panel(self)

		# Configure Plotting Panel
		self.plot_pnl = wx.Panel(self)
		self.plot_pnl.SetBackgroundColour(wx.BLACK)
		self.fig = Figure()
		self.gs = GridSpec(HEIGHT,WIDTH) # (height,width)
		plot_area = self.gs[:-1,SIDEBAR_WIDTH:]
		metric_areas = [self.gs[x,:SIDEBAR_WIDTH] for x in range(HEIGHT)]
		self.canvas = FigureCanvas(self.plot_pnl, -1, self.fig)
		self.axes = self.fig.add_axes([0, 0, 1, 1], )
		self.plot_axes = self.fig.add_subplot(plot_area, projection = '3d')
		self.metric_axes = [self.fig.add_subplot(metric_areas[i]) for i in range(HEIGHT)]
		self.metric_xlines =  [None for i in range(HEIGHT)]
		self.trail_opacity=0.7

		# Configure Sphere plotting on plot_pnl
		self.sphere_enabled=True
		self.sphere_line_collection = None
		self.sphere_opacity=0.9


		# Configure Control Panel
		self.control_pnl = wx.Panel(self)
		self.time_slider = wx.Slider(self.control_pnl, value = 0, minValue = 0, maxValue = 1)
		self.pause_btn = wx.Button(self.control_pnl, label = "Pause")
		self.play_btn = wx.Button(self.control_pnl, label = "Play")
		self.faster_btn = wx.Button(self.control_pnl, label = "Rate++")
		self.slower_btn = wx.Button(self.control_pnl, label = "Rate--")
		self.slider2 = wx.Slider(self.control_pnl, value = 1, minValue = 0, maxValue = 100, size = (120, -1))
		self.sphere_chk = wx.CheckBox(self.control_pnl, label = "Sphere")

		self.Bind(wx.EVT_SCROLL, self.on_time_slider, self.time_slider)
		self.Bind(wx.EVT_BUTTON, self.on_pause_btn, self.pause_btn)
		self.Bind(wx.EVT_UPDATE_UI, self.on_update_pause_btn, self.pause_btn)
		self.Bind(wx.EVT_BUTTON, self.on_play_btn, self.play_btn)
		self.Bind(wx.EVT_BUTTON, self.on_faster_btn, self.faster_btn)
		self.Bind(wx.EVT_BUTTON, self.on_slower_btn, self.slower_btn)
		self.Bind(wx.EVT_CHECKBOX, self.on_sphere_chk, self.sphere_chk)

		vbox = wx.BoxSizer(wx.VERTICAL)
		hbox1 = wx.BoxSizer(wx.HORIZONTAL)
		hbox2 = wx.BoxSizer(wx.HORIZONTAL)

		hbox1.Add(self.time_slider, proportion = 1)
		hbox2.Add(self.pause_btn)
		hbox2.Add(self.play_btn, flag = wx.RIGHT, border = 5)
		hbox2.Add(self.faster_btn, flag = wx.LEFT, border = 5)
		hbox2.Add(self.slower_btn)
		hbox2.Add(self.slider2, flag = wx.TOP | wx.LEFT, border = 5)
		hbox2.Add(self.sphere_chk)

		vbox.Add(hbox1, flag = wx.EXPAND | wx.BOTTOM, border = 10)
		vbox.Add(hbox2, proportion = 1, flag = wx.EXPAND)
		self.control_pnl.SetSizer(vbox)

		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.sizer.Add(self.plot_pnl, proportion = 1, flag = wx.EXPAND)
		self.sizer.Add(self.control_pnl, flag = wx.EXPAND | wx.BOTTOM | wx.TOP, border = 10)

		self.Bind(wx.EVT_SIZE, self.on_resize)
		self.Bind(wx.EVT_IDLE, self.on_idle)

		self.SetMinSize((800, 600))
		self.CreateStatusBar()
		self.SetSizer(self.sizer)

		self.SetSize((350, 200))
		self.SetTitle('Player')
		self.Centre()
		self.Show(True)

		if self.args.data_file is not None:
			if os.path.exists(self.args.data_file):
				self.log.info("Loading data from %s" % self.args.data_file)
				self.load_data(self.args.data_file)
			else:
				self.log.error("File Not Found: %s" % self.args.data_file)

	def CreateMenuBar(self):
		menubar = wx.MenuBar()
		filem = wx.Menu()
		play = wx.Menu()
		view = wx.Menu()
		tools = wx.Menu()
		favorites = wx.Menu()
		help = wx.Menu()

		openm = filem.Append(wx.NewId(), '&Open \t Ctrl+o', 'Open a datafile')
		self.Bind(wx.EVT_MENU, self.on_open, openm)
		exitm = filem.Append(wx.NewId(), '&Quit', 'Quit application')
		self.Bind(wx.EVT_MENU, self.on_close, exitm)

		menubar.Append(filem, '&File')
		menubar.Append(play, '&Play')
		menubar.Append(view, '&View')
		menubar.Append(tools, '&Tools')
		menubar.Append(favorites, 'F&avorites')
		menubar.Append(help, '&Help')

		self.accel_tbl = wx.AcceleratorTable(
			[(wx.ACCEL_CTRL, ord('o'), openm.GetId())]
		)
		self.SetAcceleratorTable(self.accel_tbl)

		self.SetMenuBar(menubar)

	def init_plot(self):
		# Find initial display state for viewport
		self.lines = [self.plot_axes.plot(xs, ys, zs, alpha=self.trail_opacity)[0] for xs, ys, zs in self.data.p]
		for n, line in enumerate(self.lines):
			line.set_label(self.data.names[n])


		# Initialise Sphere data anyway
		stddevrange = self.data.stddev_range()
		self.log.debug("STD: MIN: %f, MAX: %f"%(min(stddevrange),max(stddevrange)))
		self.plot_lognorm=Normalize(vmin=min(stddevrange), vmax=max(stddevrange))
		self.log.debug("%s"%str([ self.plot_lognorm(i) for i in range( int(min(stddevrange)), 100, 5)]))
		self.plot_sphere_cm = cm.Spectral_r

		# Initialse Positional Plot
		shape = self.data.environment
		self.plot_axes.set_title("Tracking overview of %s" % self.data.title)
		self.plot_axes.set_xlim3d((0, shape[0]))
		self.plot_axes.set_ylim3d((0, shape[1]))
		self.plot_axes.set_zlim3d((0, shape[2]))
		self.plot_axes.set_xlabel('X')
		self.plot_axes.set_ylabel('Y')
		self.plot_axes.set_zlabel('Z')


		# Initialise StdDev Plot
		self.metric_axes[0].plot(stddevrange)
		self.metric_axes[0].set_ylabel("StdDev")
		self.metric_axes[0].get_xaxis().set_visible(False)




	def redraw_plot(self):
		###
		# MAIN PLOT AREA
		###
		for n, line in enumerate(self.lines):
			(xs, ys, zs) = self.data.trail_of(n, self.t)
			line.set_data(xs, ys)
			line.set_3d_properties(zs)
			line.set_label(self.data.names[n])

		###
		# SPHERE OVERLAY TO MAIN PLOT AREA
		###
		if self.sphere_enabled:
			(x,y,z),r,s = self.data.sphere_of_positions_with_stddev(self.t)
			xs,ys,zs = self.sphere(x,y,z,r)
			colorval = self.plot_lognorm(s)
			self.log.info("Average position: %s, Color: %s[%s], StdDev: %s"%(str((x,y,z)),str(self.plot_sphere_cm(colorval)),str(colorval),str(s)))

			self._remove_sphere()
			self.sphere_line_collection = self.plot_axes.plot_wireframe(xs,ys,zs,
				alpha=self.sphere_opacity,
				color=self.plot_sphere_cm(colorval)
			)

		###
		# METRIC UPDATES
		###
		try:
			self.metric_xlines[0].remove()
		except AttributeError as e:
			self.log.debug("Hopefully nothing: %s"%str(e))
		self.metric_xlines[0]=self.metric_axes[0].axvline(x=self.t, color='r', linestyle=':')
		self.canvas.draw()



	def resize_panel(self):
		plot_size = self.sizer.GetChildren()[0].GetSize()
		self.plot_pnl.SetSize(plot_size)
		self.canvas.SetSize(plot_size)
		self.fig.set_size_inches(float(plot_size[0]) / self.fig.get_dpi(),
								 float(plot_size[0]) / self.fig.get_dpi()
		)

	def move_T(self, delta_t = None):
		""" Seek the visual plot by delta_t while doing bounds checking and redraw

		: param delta_t: Positive or negative time shift from current t. If None use t_d
		: type delta_t: int

		"""

		self.t += delta_t if delta_t is not None else self.d_t
		# If trying to go over the end, don't, and either stop or loop
		if self.t >= self.data.tmax-1:
			if self.args.loop:
				self.log.debug("Looping")
				self.t=0
			else:
				self.log.debug("End Of The Line")
				self.paused=True
				self.t=self.data.tmax-1
		if self.t < 0:
			self.log.debug("Tried to reverse, pausing")
			self.paused=True
			self.t=0
		self.time_slider.SetValue(self.t)
		self.redraw_plot()


	def load_data(self, data_file):
		""" Load an external data file for plotting and navigation

		:param data_file: Aietes generated data for processing
		:type data_file: str

		Configures Plot area and adjusts control area
		"""

		self.data = DataPackage(data_file)
		self.init_plot()
		self.log.debug("Successfully loaded data from %s, containing %d nodes over %d seconds" % (
			self.data.title,
			self.data.n,
			self.data.tmax
			)
		)
		self.time_slider.SetRange(0, self.data.tmax)
		self.time_slider.SetValue(self.data.tmax)
		self.d_t = int(self.data.tmax / 100)
		if self.args.autostart:
			self.paused = False

	def smartass(self, msg = None):
		self.paused = True
		if self.telling_off is False:
			self.telling_off = True
			dial = wx.MessageDialog(None, 'Stop it Smartass!' if msg is None else str(msg), "Smartass",
									wx.OK | wx.ICON_EXCLAMATION)
			dial.ShowModal()
			self.telling_off = False


	####
	# Event Operations
	####

	def on_close(self, event):
		dlg = wx.MessageDialog(self,
							   "Do you really want to close this application?",
							   "Confirm Exit", wx.OK | wx.CANCEL | wx.ICON_QUESTION)
		result = dlg.ShowModal()
		dlg.Destroy()
		if result == wx.ID_OK:
			self.Destroy()

	def on_open(self, event):
		dlg = wx.FileDialog(
			self, message = "Select a DataPackage",
			defaultDir = os.getcwd(),
			wildcard = "*.npz",
			style = wx.OPEN | wx.CHANGE_DIR
		)

		if dlg.ShowModal() == wx.ID_OK:
			data_path = dlg.GetPaths()
			if (len(data_path) > 1):
				self.log.warn("Too many paths given, only taking the first anyway")
			data_path = data_path[0]
			self.load_data(data_path)

	def on_pause_btn(self, event):
		self.paused = not self.paused

	def on_update_pause_btn(self, event):
		self.pause_btn.SetLabel("Resume" if self.paused else "Pause")

	def on_play_btn(self, event):
		self.t=0
		self.paused = False
		pass

	def on_faster_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 1.1), self.data.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)

	def on_slower_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 0.9), self.data.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)



	def on_time_slider(self, event):
		event.Skip()
		self.t = self.time_slider.GetValue()
		self.log.debug("Slider: Setting time to %d" % self.t)
		wx.CallAfter(self.redraw_plot)
		pass

	def on_resize(self, event):
		event.Skip()
		wx.CallAfter(self.resize_panel)

	def on_idle(self, event):
		event.Skip()
		if not self.paused:
			self.move_T()

	###
	# Metric Selection Tools
	###
	def on_sphere_chk(self, event):
		if not self.sphere_chk.IsChecked():
			self._remove_sphere()
			self.log.debug("Sphere Overlay Disabled")
			self.sphere_enabled = False
		else:
			self.log.debug("Sphere Overlay Enabled")
			self.sphere_enabled = True
	####
	# Plotting Tools
	###
	def sphere(self,x,y,z,r=1.0):
		"""
		Returns a sphere definition tuple (xs,ys,zs) for use with plot_wireframe
		"""
		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
		xs=(r*np.cos(u)*np.sin(v))+x
		ys=(r*np.sin(u)*np.sin(v))+y
		zs=(r*np.cos(v))+z

		return (xs,ys,zs)

	def _remove_sphere(self):
		if isinstance(self.sphere_line_collection, Line3DCollection) \
		and self.sphere_line_collection in self.plot_axes.collections:

			self.plot_axes.collections.remove(self.sphere_line_collection)


def main():
	app = wx.PySimpleApp()
	app.frame = EphyraFrame(None)
	app.frame.Show()
	app.MainLoop()

if __name__ == '__main__':
	main()
