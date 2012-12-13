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
import collections
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

matplotlib.rcParams.update({'font.size': 8})

import numpy as np


_ROOT = os.path.abspath(os.path.dirname(__file__))
WIDTH,HEIGHT = 8,6
SIDEBAR_WIDTH = 2

class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		FancyArrowPatch.draw(self, renderer)

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
		parser.add_argument('-v', '--verbose',
			dest = 'verbose', action = 'store_true', default = False,
			help = 'Verbose Debugging Information'
		)
		parser.add_argument('-n', '--new-simulation',
			dest = 'newsim', action = 'store_true', default = False,
			help = 'Generate a new simulation from default'
		)

		self.args = parser.parse_args()

		self.paused = True
		self.simulating = False
		self.sim_ready = False
		self.telling_off = False # Used for Smartass Control
		self.t = 0
		self.tmax = None
		self.sim_tmax = None
		self.d_t = 1

		self.CreateMenuBar()
		panel = wx.Panel(self)

		# Configure Plotting Panel
		self.plot_pnl = wx.Panel(self)
		self.fig = Figure()
		self.gs = GridSpec(HEIGHT,WIDTH) # (height,width)
		plot_area = self.gs[:-1,SIDEBAR_WIDTH:]
		metric_areas = [self.gs[x,:SIDEBAR_WIDTH] for x in range(HEIGHT)]
		self.canvas = FigureCanvas(self.plot_pnl, -1, self.fig)

		self.axes = self.fig.add_axes([0, 0, 1, 1], )
		self.plot_axes = self.fig.add_subplot(plot_area, projection = '3d')
		self.metric_axes = [self.fig.add_subplot(metric_areas[i]) for i in range(HEIGHT)]
		self.metric_plots = []
		self.metric_xlines =  [None for i in range(HEIGHT)]
		self.trail_opacity=0.7
		self.trail = 100

		# Configure Sphere plotting on plot_pnl
		self.sphere_enabled=True
		self.sphere_line_collection = None
		self.sphere_opacity = 0.9

		#Configure Vector Plotting on Plot_pnl
		self.node_vector_enabled=True
		self.fleet_vector_enabled=True
		self.node_vector_collections = None
		self.fleet_vector_collection = None
		self.vector_opacity = 0.9


		# Configure Control Panel
		self.control_pnl = wx.Panel(self)
		self.time_slider = wx.Slider(self.control_pnl, value = 0, minValue = 0, maxValue = 1)
		self.pause_btn = wx.Button(self.control_pnl, label = "Pause")
		self.play_btn = wx.Button(self.control_pnl, label = "Play")
		self.faster_btn = wx.Button(self.control_pnl, label = "Rate++")
		self.slower_btn = wx.Button(self.control_pnl, label = "Rate--")
		self.trail_slider = wx.Slider(self.control_pnl, value = self.trail, minValue = 0, maxValue = 100, size = (120, -1))

		self.Bind(wx.EVT_SCROLL, self.on_time_slider, self.time_slider)
		self.Bind(wx.EVT_SCROLL, self.on_trail_slider, self.trail_slider)
		self.Bind(wx.EVT_BUTTON, self.on_pause_btn, self.pause_btn)
		self.Bind(wx.EVT_UPDATE_UI, self.on_update_pause_btn, self.pause_btn)
		self.Bind(wx.EVT_BUTTON, self.on_play_btn, self.play_btn)
		self.Bind(wx.EVT_BUTTON, self.on_faster_btn, self.faster_btn)
		self.Bind(wx.EVT_BUTTON, self.on_slower_btn, self.slower_btn)



		vbox = wx.BoxSizer(wx.VERTICAL)
		hbox1 = wx.BoxSizer(wx.HORIZONTAL)
		hbox2 = wx.BoxSizer(wx.HORIZONTAL)

		hbox1.Add(self.time_slider, proportion = 1)
		hbox2.Add(self.pause_btn)
		hbox2.Add(self.play_btn, flag = wx.RIGHT, border = 5)
		hbox2.Add(self.faster_btn, flag = wx.LEFT, border = 5)
		hbox2.Add(self.slower_btn)
		hbox2.Add(self.trail_slider, flag = wx.TOP | wx.LEFT, border = 5)

		#Metric Buttons
		self.sphere_chk = wx.CheckBox(self.control_pnl, label = "Sphere")
		self.vector_chk = wx.CheckBox(self.control_pnl, label = "Vector")
		self.sphere_chk.SetValue(self.sphere_enabled)
		self.vector_chk.SetValue(self.node_vector_enabled)
		self.Bind(wx.EVT_CHECKBOX, self.on_sphere_chk, self.sphere_chk)
		self.Bind(wx.EVT_CHECKBOX, self.on_vector_chk, self.vector_chk)
		hbox2.Add(self.sphere_chk)
		hbox2.Add(self.vector_chk)

		vbox.Add(hbox1, flag = wx.EXPAND | wx.BOTTOM, border = 10)
		vbox.Add(hbox2, proportion = 1, flag = wx.EXPAND)
		self.control_pnl.SetSizer(vbox)

		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.sizer.Add(self.plot_pnl, proportion = 1, flag = wx.EXPAND)
		self.sizer.Add(self.control_pnl, flag = wx.EXPAND | wx.BOTTOM | wx.TOP)

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
				self.smartass("File Not Found!")
				self.on_open(None)

		if self.args.newsim:
			self.new_simulation()

	def CreateMenuBar(self):
		menubar = wx.MenuBar()
		filem = wx.Menu()
		play = wx.Menu()
		view = wx.Menu()
		tools = wx.Menu()
		favorites = wx.Menu()
		help = wx.Menu()

		newm = filem.Append(wx.NewId(), '&New \t Ctrl+n', 'Generate a new Simulation')
		openm = filem.Append(wx.NewId(), '&Open \t Ctrl+o', 'Open a datafile')
		exitm = filem.Append(wx.NewId(), '&Quit', 'Quit application')
		self.Bind(wx.EVT_MENU, self.on_new, newm)
		self.Bind(wx.EVT_MENU, self.on_open, openm)
		self.Bind(wx.EVT_MENU, self.on_close, exitm)
		menubar.Append(filem, '&File')

		menubar.Append(play, '&Play')
		node_selectm = filem.Append(wx.NewId(), '&Select Displayed Nodes \t Ctrl+N', 'Limit the analytics display to selected nodes')
		self.Bind(wx.EVT_MENU, self.on_node_select, node_selectm)
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

		# Metrics
		Metric = collections.namedtuple('Metric', ['label', 'data'], verbose=False)
		metrics=[]
		metrics.append(Metric("Dist Stddev",self.data.distance_from_average_stddev_range()))
		metrics.append(Metric("InterN-Dist",[self.data.inter_distance_average(time) for time in range(int(self.data.tmax))])),
		metrics.append(Metric("Head-StdDev",self.data.heading_stddev_range())),
		metrics.append(Metric("Head-MagAvg",self.data.heading_mag_range())),
		metrics.append(Metric("Head-AvgMag",[map(np.linalg.norm,self.data.heading_slice(time)) for time in range(int(self.data.tmax))])),
		metrics.append(Metric("Avg",[np.asarray(self.data.distances_from_average_at(time)) for time in range(int(self.data.tmax))]))

		# Start off with all nodes displayed
		self.displayed_nodes = np.empty(self.data.n, dtype=bool)
		self.displayed_nodes.fill(False)

		# Find initial display state for viewport
		self.lines = [self.plot_axes.plot(xs, ys, zs, alpha=self.trail_opacity)[0] for xs, ys, zs in self.data.p]
		for n, line in enumerate(self.lines):
			line.set_label(self.data.names[n])

		position_stddevrange = self.data.distance_from_average_stddev_range()
		heading_stddevrange = self.data.heading_stddev_range()
		heading_magrange = self.data.heading_mag_range()

		assert len(heading_stddevrange) == self.data.tmax, "H-%s"%str(heading_stddevrange)

		self.plot_head_mag_norm=Normalize(vmin=min(heading_magrange), vmax=max(heading_magrange))
		self.plot_pos_stddev_norm=Normalize(vmin=min(position_stddevrange), vmax=max(position_stddevrange))

		# Initialise Sphere data anyway
		self.log.debug("STD: MIN: %f, MAX: %f"%(min(position_stddevrange),max(position_stddevrange)))
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

		for i,metric in enumerate(metrics):
			self.metric_plots.append(PerNodeGraph_Axes(axes=self.metric_axes[i],
			                                           data=metric.data,
			                                           label=metric.label)
			)


		self.plot_analysis()


	def  plot_analysis(self):

		for i,plot in enumerate(self.metric_plots):
			if isinstance(plot.data[0],float):
				#one dimension, no need for wanted filter
				self.metric_axes[i]=plot.plot()
			else:
				self.metric_axes[i]=plot.plot(wanted=np.asarray(self.displayed_nodes))



	def new_simulation(self):
		from aietes import Simulation
		self.simulation = Simulation()
		sim_info=self.simulation.prepare(waits=True)
		self.tmax=sim_info['sim_time']
		self.simulating = True
		self.sim_ready = False
		self.sim_tmax=0
		self.simulation.simulate(callback=self.simulation_cb)

	def simulation_cb(self):
		"""
		Call back function used by SimulationStep if doing real time simulation
		Will 'export' DataPackage data from the running simulation up to the requested time (self.t)
		"""
		#TODO expand this to allow 'reimagining'
		# Reload data when fleet is computed fully, simulation has gone for a while, and
		#     when user has navigated beyond the current sim_tmax

		if self.t<=self.sim_tmax and not self.simulation.waiting and self.simulation.now()>10:

			(p,v,names,environment)=self.simulation.currentState()
			self.data=DataPackage(p=p,v=v,names=names,environment=environment)
			self.sim_tmax=self.simulation.now()-1
			self.simulation.waiting = False
			self.log.info("Reloading at %d from tmax:%d and t:%d"%(self.simulation.now(),self.sim_tmax,self.t))
			self.reload_data()
			if not hasattr(self,'lines'):
				self.log.info("First Plot")
				self.init_plot()
			self.sim_ready=True
			#self.paused = False
			while self.t<=self.sim_tmax:
				self.log.info("Yielding %d,%d"%(self.t, self.sim_tmax))
				wx.Yield()
				self.log.info("Returned From Yield %d,%d"%(self.t, self.sim_tmax))

		else:
			self.log.info("Continuing at %d from tmax:%d and t:%d"%(self.simulation.now(),self.sim_tmax,self.t))
			self.simulation.waiting = False
			self.sim_ready = False
			self.paused = True
			return # Continue Simulating



	def redraw_plot(self):
		wx.Yield()
		###
		# MAIN PLOT AREA
		###
		for n, line in enumerate(self.lines):
			(xs, ys, zs) = self.data.trail_of(n, self.t, self.trail)
			line.set_data(xs, ys)
			line.set_3d_properties(zs)
			line.set_label(self.data.names[n])

		###
		# SPHERE OVERLAY TO MAIN PLOT AREA
		###
		if self.sphere_enabled:
			(x,y,z),r,s = self.data.sphere_of_positions_with_stddev(self.t)
			xs,ys,zs = self.sphere(x,y,z,r)
			colorval = self.plot_pos_stddev_norm(s)
			if self.args.verbose: self.log.debug("Average position: %s, Color: %s[%s], StdDev: %s"%(str((x,y,z)),str(self.plot_sphere_cm(colorval)),str(colorval),str(s)))

			self._remove_sphere()
			self.sphere_line_collection = self.plot_axes.plot_wireframe(xs,ys,zs,
				alpha=self.sphere_opacity,
				color=self.plot_sphere_cm(colorval)
			)

		###
		# VECTOR OVERLAYS TO MAIN PLOT AREA
		###
		# Arrow3D
		if self.node_vector_enabled:
			self._remove_vectors()
			for node in range(self.data.n):
				position = self.data.position_of(node, self.t)
				heading = self.data.heading_of(node,self.t)
				mag = np.linalg.norm(np.asarray(heading))
				colorval = self.plot_head_mag_norm(mag)
				if self.args.verbose: self.log.debug("Average heading: %s, Color: [%s], Speed: %s"%(str(heading),str(colorval),str(mag)))

				xs,ys,zs = zip(position,np.add(position,(np.asarray(heading)*50)))
				self.node_vector_collections[node] = Arrow3D(
					xs,ys,zs,
					mutation_scale=2, lw=1,
					arrowstyle="-|>", color=self.plot_sphere_cm(colorval), alpha=self.vector_opacity
				)
				self.plot_axes.add_artist(
					self.node_vector_collections[node],
				)


		###
		# METRIC UPDATES
		###
		for x in range(HEIGHT):
			try:
				self.metric_xlines[x].remove()
			except AttributeError as e:
				self.log.debug("Hopefully nothing: %s"%str(e))
			except ValueError as e:
				self.log.debug("Hopefully a different nothing: %s"%str(e))
			self.metric_xlines[x]=self.metric_axes[x].axvline(x=self.t, color='r', linestyle=':')
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
		if self.t > self.tmax:
			if self.args.loop:
				self.log.debug("Looping")
				self.t=0
			else:
				self.log.debug("End Of The Line")
				self.paused=True
				self.t=self.tmax
		if self.t < 0:
			self.log.debug("Tried to reverse, pausing")
			self.paused=True
			self.t=0
		if self.simulating and self.t < self.sim_tmax:
			wx.Yield()
		else:
			self.time_slider.SetValue(self.t)
			self.redraw_plot()


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
		self.init_plot()
		self.log.debug("Successfully loaded data from %s, containing %d nodes over %d seconds" % (
		self.data.title,
		self.data.n,
		self.data.tmax
		)
		)
		self.tmax = self.data.tmax-1
		self.time_slider.SetRange(0, self.tmax)
		self.time_slider.SetValue(0)
		self.d_t = int((self.tmax+1) / 100)

		if self.args.autostart:
			self.paused = False


	def reload_data(self):
		self.init_plot()
		self.log.debug("Successfully reloaded data from %s, containing %d nodes over %d seconds" % (
		self.data.title,
		self.data.n,
		self.data.tmax
		)
		)
		self.time_slider.SetRange(0, self.tmax)
		self.d_t = int((self.tmax+1) / 100)
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

	def on_new(self, event):
		dlg = wx.MessageDialog(self,
			message="This will start a new simulation using the SimulationStep system to generate results in 'real' time and will be fucking slow",
			style= wx.OK | wx.CANCEL | wx.ICON_EXCLAMATION
		)
		result = dlg.ShowModal()
		dlg.Destroy()
		if result == wx.ID_OK:
			self.new_simulation()

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

	def on_faster_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 1.1), self.data.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)

	def on_slower_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 0.9), self.data.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)


	def on_time_slider(self, event):
		self.t = self.time_slider.GetValue()
		self.log.debug("Slider: Setting time to %d" % self.t)
		wx.CallAfter(self.redraw_plot)

	def on_resize(self, event):
		event.Skip()
		wx.CallAfter(self.resize_panel)

	def on_idle(self, event):
		wx.Yield()
		if not self.paused:
			self.move_T()

	###
	# Display Selection Tools
	###
	def on_sphere_chk(self, event):
		if not self.sphere_chk.IsChecked():
			self._remove_sphere()
			self.log.debug("Sphere Overlay Disabled")
			self.sphere_enabled = False
		else:
			self.log.debug("Sphere Overlay Enabled")
			self.sphere_enabled = True
		wx.CallAfter(self.redraw_plot)

	def on_vector_chk(self, event):
		if not self.vector_chk.IsChecked():
			self._remove_vectors()
			self.log.debug("Vector Overlay Disabled")
			self.node_vector_enabled = False
		else:
			self.log.debug("Vector Overlay Enabled")
			self.node_vector_enabled = True
		wx.CallAfter(self.redraw_plot)

	def on_trail_slider(self, event):
		event.Skip()
		norm_trail = self.trail_slider.GetValue()
		self.trail = int(norm_trail*(self.data.tmax/100.0))
		self.log.debug("Slider: Setting trail to %d" % self.trail)
		wx.CallAfter(self.redraw_plot)

	def on_node_select(self, event):
		"""
		Based on the wxPython demo - opens the MultiChoiceDialog
		and sets that selection as the node entries to be
		displayed
		"""
		lst = list(self.data.names)
		dlg = wx.MultiChoiceDialog( self,
		                            "Select nodes",
		                            "wx.MultiChoiceDialog", lst)

		selections = [ i for i in range(self.data.n) if self.displayed_nodes[i]]
		print selections
		dlg.SetSelections(selections)

		if (dlg.ShowModal() == wx.ID_OK):
			selections = dlg.GetSelections()
			strings = [lst[x] for x in selections]
			print "You chose:" + str(strings)
			self.displayed_nodes.fill(False)
			for checked in selections:
				self.displayed_nodes[checked]=True

			self.plot_analysis()

		wx.CallAfter(self.redraw_plot)


####
	# Plotting Tools
	###
	def sphere(self,x,y,z,r=1.0):
		"""
		Returns a sphere definition tuple (xs,ys,zs) for use with plot_wireframe
		"""
		u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
		xs=(r*np.cos(u)*np.sin(v))+x
		ys=(r*np.sin(u)*np.sin(v))+y
		zs=(r*np.cos(v))+z

		return (xs,ys,zs)

	def _remove_sphere(self):
		if isinstance(self.sphere_line_collection, Line3DCollection) \
		and self.sphere_line_collection in self.plot_axes.collections:

			self.plot_axes.collections.remove(self.sphere_line_collection)

	def _remove_vectors(self):
		if self.node_vector_collections is not None:
			for arrow in self.node_vector_collections:
				if arrow is not None:
					arrow.remove()

		self.node_vector_collections = [ None for i in range(self.data.n)]

class PerNodeGraph_Axes():
	# Assumes that data is constant and only needs to be selected per node
	def __init__(self, axes, data, *args, **kw):
		self.ax = axes
		self.data = data
		self.ax.set_ylabel(kw.get('label',"UNDEFINED"))
		self.ax.get_xaxis().set_visible(False)

	def plot(self, wanted=None):
		"""
		Update the Plot based on 'new' wanted data
		"""
		self.ax.clear()
		if wanted is None:
			self.ax.plot(self.data, alpha=0.3)
		else:
			logging.info("Printing %s with Wanted:%s"%(self,wanted))
			self.ax.plot(np.asarray(self.data)[wanted,:],alpha=0.3)
		return self.ax

	def __repr__(self):
		return "PerNodeGraph_Axes: %s with %s values"%(
			self.ax.get_ylabel(),
		    len(self.data)
		)


def main():
	app = wx.PySimpleApp()
	app.frame = EphyraFrame(None)
	app.frame.Show()
	app.MainLoop()

if __name__ == '__main__':
	main()
