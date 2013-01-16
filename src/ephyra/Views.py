__author__ = 'andrewbolster'

import wx
import os
import logging
import argparse
import traceback, sys

import numpy as np

import matplotlib
matplotlib.use('WXAgg')

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib import cm

matplotlib.rcParams.update({'font.size': 8})

WIDTH, HEIGHT = 8, 6
SIDEBAR_WIDTH = 2

class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
		FancyArrowPatch.draw(self, renderer)


class Notebook(wx.Frame):
	def __init__(self, controller, *args, **kw):
		super(Notebook, self).__init__(*args, **kw)
		self.log = logging.getLogger(self.__module__)
		self.ctl = controller
		parser = argparse.ArgumentParser(description = "GUI Simulation and Analysis Suite for the Aietes framework")

		parser.add_argument('-o', '--open',
		                    dest = 'data_file', action = 'store', default = None,
		                    metavar = 'XXX.npz',
		                    help = 'Aietes DataPackage to be analysed'
		)
		parser.add_argument('-a', '--autostart',
		                    dest = 'autostart', action = 'store_true', default = False,
		                    help = 'Automatically launch animation on loading'
		)
		parser.add_argument('-x', '--autoexit',
		                    dest = 'autoexit', action = 'store_true', default = False,
		                    help = 'Automatically exit after animation'
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

		# Create a panel and a Notebook on the Panel
		self.CreateMenuBar()
		self.status_bar = self.CreateStatusBar()
		self.status_bar.SetFieldsCount(3)

		self.Bind(wx.EVT_SIZE, self.on_resize)
		self.Bind(wx.EVT_IDLE, self.on_idle)

		p = wx.Panel(self)
		self.nb = wx.Notebook(p)

		pages = [VisualNavigator, Configurator, Simulator]

		for page in pages:
			self.nb.AddPage(page(self.nb, self), page.__class__.__name__)

		sizer = wx.BoxSizer()
		sizer.Add(self.nb, 1, wx.EXPAND)
		p.SetSizer(sizer)


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

		menubar.Append(view, '&View')

		menubar.Append(tools, '&Tools')

		menubar.Append(favorites, 'F&avorites')

		menubar.Append(help, '&Help')

		self.accel_tbl = wx.AcceleratorTable(
			[(wx.ACCEL_CTRL, ord('o'), openm.GetId())]
		)
		self.SetAcceleratorTable(self.accel_tbl)

		self.SetMenuBar(menubar)


	####
	# Window Events
	####
	def on_close(self, event):
		dlg = wx.MessageDialog(self,
		                       "Do you really want to close this application?",
		                       "Confirm Exit", wx.OK | wx.CANCEL | wx.ICON_QUESTION)
		result = dlg.ShowModal()
		dlg.Destroy()
		if result == wx.ID_OK:
			self.Destroy()

	def on_resize(self, event):
		event.Skip()
		wx.CallAfter(self.resize_panel)

	####
	# File Events
	####
	def on_new(self, event):
		dlg = wx.MessageDialog(self,
		                       message = "This will start a new simulation using the SimulationStep system to generate results in 'real' time and will be fucking slow",
		                       style = wx.OK | wx.CANCEL | wx.ICON_EXCLAMATION
		)
		result = dlg.ShowModal()
		dlg.Destroy()
		if result == wx.ID_OK:
			self.ctl.new_simulation()

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
			self.ctl.load_data_file(data_path)

	###
	# Status Management
	###
	def update_status(self, msg):
		self.status_bar.SetStatusText(msg, number = 0)
		if msg is not "Idle":
			self.log.info(msg)

	def update_timing(self):
		self.status_bar.SetStatusText("%s:%d/%d" % ("SIM" if self.simulating else "REC", self.t, self.tmax), number = 2)

	def on_idle(self, event):
		try:
			self.nb.GetCurrentPage().on_idle(event)
		except:
			traceback.print_exc(file=sys.stdout)
			self.Destroy()

	def on_resize(self, event):
		try:
			self.nb.GetCurrentPage().on_resize(event)
		except:
			traceback.print_exc(file=sys.stdout)
			self.Destroy()

class Configurator(wx.Panel):
	def __init__(self, parent, frame, *args, **kw):
		super(wx.Panel, self).__init__(parent, *args, **kw)

		self.Show(True)

class Simulator(wx.Panel):
	def __init__(self, parent, frame, *args, **kw):
		super(wx.Panel, self).__init__(parent, *args, **kw)
		self.Show(True)

class VisualNavigator(wx.Panel):
	def __init__(self, parent, frame, *args, **kw):
		super(wx.Panel, self).__init__(parent, *args, **kw)
		self.log = frame.log.getChild(self.__class__.__name__)
		
		self.frame = frame

		# Timing and playback defaults
		self.paused = None
		self.t = 0
		self.tmax = None # Not always the same as the data tmax eg simulating
		self.d_t = 1

		# Configure Plotting Panel
		self.plot_pnl = wx.Panel(self)
		self.fig = Figure()
		self.gs = GridSpec(HEIGHT, WIDTH) # (height,width)


		self.trail_opacity = 0.7
		self.trail = 100

		# Configure Sphere plotting on plot_pnl
		self.sphere_enabled = True
		self.sphere_line_collection = None
		self.sphere_opacity = 0.9

		#Configure Vector Plotting on Plot_pnl
		self.node_vector_enabled = True
		self.fleet_vector_enabled = True
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
		self.trail_slider = wx.Slider(self.control_pnl, value = self.trail, minValue = 0, maxValue = 100,
		                              size = (120, -1))

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
		self.sphere_chk.SetValue(self.sphere_enabled)
		self.Bind(wx.EVT_CHECKBOX, self.on_sphere_chk, self.sphere_chk)
		hbox2.Add(self.sphere_chk)

		self.vector_chk = wx.CheckBox(self.control_pnl, label = "Vector")
		self.vector_chk.SetValue(self.node_vector_enabled)
		self.Bind(wx.EVT_CHECKBOX, self.on_vector_chk, self.vector_chk)
		hbox2.Add(self.vector_chk)

		vbox.Add(hbox1, flag = wx.EXPAND | wx.BOTTOM, border = 10)
		vbox.Add(hbox2, proportion = 1, flag = wx.EXPAND)
		self.control_pnl.SetSizer(vbox)

		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.sizer.Add(self.plot_pnl, proportion = 1, flag = wx.EXPAND)
		self.sizer.Add(self.control_pnl, flag = wx.EXPAND | wx.BOTTOM | wx.TOP)

		self.SetMinSize((800, 600))

		self.SetSizer(self.sizer)
		self.Show(True)

	####
	# 3D Plot Handling Functions
	####

	def initialise_3d_plot(self):

		metric_areas = [self.gs[x, :SIDEBAR_WIDTH] for x in range(HEIGHT)]
		self.canvas = FigureCanvas(self.plot_pnl, -1, self.fig)

		self.axes = self.fig.add_axes([0, 0, 1, 1], )

		self.metric_axes = [self.fig.add_subplot(metric_areas[i]) for i in range(HEIGHT)]
		for ax in self.metric_axes:
			ax.autoscale_view(scalex = False, tight = True)

		self.metric_xlines = [None for i in range(HEIGHT)]
		plot_area = self.gs[:-1, SIDEBAR_WIDTH:]
		self.plot_axes = self.fig.add_subplot(plot_area, projection = '3d')

		# Start off with all nodes displayed
		self.displayed_nodes = np.empty(self.data.n, dtype = bool)
		self.displayed_nodes.fill(True)

		# Initialise Sphere data anyway
		self.plot_sphere_cm = cm.Spectral_r

		# Initialse Positional Plot
		shape = self.ctl.get_extent()
		self.plot_axes.set_title("Tracking overview of %s" % self.data.title)
		self.plot_axes.set_xlim3d((0, shape[0]))
		self.plot_axes.set_ylim3d((0, shape[1]))
		self.plot_axes.set_zlim3d((0, shape[2]))
		self.plot_axes.set_xlabel('X')
		self.plot_axes.set_ylabel('Y')
		self.plot_axes.set_zlabel('Z')

		#Initialise Data-Based timings.
		self.tmax = self.ctl.get_final_tmax() - 1
		self.time_slider.SetValue(0)
		self.d_t = int((self.tmax + 1) / 100)

		self.update_status("Plotting %d metrics" % len(self.metrics))

	def redraw_page(self, t = None):
		###
		# Update Time!
		###
		if t is not None:
			self.t = t
			del t

		###
		# MAIN PLOT AREA
		###
		for n, line in enumerate(self.lines):
			(xs, ys, zs) = self.data.trail_of(n, self.t, self.trail)

			line.set_data(xs, ys)
			line.set_3d_properties(zs)
			line.set_label(self.data.names[n])

		###
		# VECTOR OVERLAYS TO MAIN PLOT AREA
		###
		# Arrow3D
		if self.node_vector_enabled:
			self.redraw_fleet_heading_vectors()
		###
		# SPHERE OVERLAY TO MAIN PLOT AREA
		###
		if self.sphere_enabled:
			self.redraw_fleet_sphere()

		self.update_metric_charts()

		self.canvas.draw()

	def move_T(self, delta_t = None):
		""" Seek the visual plot by delta_t while doing bounds checking and redraw

		: param delta_t: Positive or negative time shift from current t. If None use t_d
		: type delta_t: int

		"""
		t = self.t
		t += delta_t if delta_t is not None else self.d_t
		# If trying to go over the end, don't, and either stop or loop
		if t > self.tmax:
			if self.frame.args.loop:
				self.log.debug("Looping")
				t = 0
			else:
				self.log.debug("End Of The Line")
				self.paused = True
				t = self.tmax
				if self.frame.args.autoexit:
					self.DestroyChildren()
					self.Destroy()

		if t < 0:
			self.log.debug("Tried to reverse, pausing")
			self.paused = True
			t = 0

		self.time_slider.SetValue(t)
		self.redraw_page(t = t)

	def resize_plots(self):
		plot_size = self.sizer.GetChildren()[0].GetSize()
		self.plot_pnl.SetSize(plot_size)
		self.canvas.SetSize(plot_size)
		self.fig.set_size_inches(float(plot_size[0]) / self.fig.get_dpi(),
		                         float(plot_size[0]) / self.fig.get_dpi()
		)

	def sphere(self, x, y, z, r = 1.0):
		"""
		Returns a sphere definition tuple (xs,ys,zs) for use with plot_wireframe
		"""
		u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
		xs = (r * np.cos(u) * np.sin(v)) + x
		ys = (r * np.sin(u) * np.sin(v)) + y
		zs = (r * np.cos(v)) + z

		return (xs, ys, zs)

	def redraw_fleet_sphere(self):
		(x, y, z), r, s = self.data.sphere_of_positions_with_stddev(self.t)
		xs, ys, zs = self.sphere(x, y, z, r)
		colorval = self.plot_pos_stddev_norm(s)
		if self.frame.args.verbose: self.log.debug("Average position: %s, Color: %s[%s], StdDev: %s" % (
		str((x, y, z)), str(self.plot_sphere_cm(colorval)), str(colorval), str(s)))

		self._remove_sphere()
		self.sphere_line_collection = self.plot_axes.plot_wireframe(xs, ys, zs,
		                                                            alpha = self.sphere_opacity,
		                                                            color = self.plot_sphere_cm(colorval)
		)

	def redraw_fleet_heading_vectors(self):
		self._remove_vectors()
		for node in range(self.data.n):
			position = self.data.position_of(node, self.t)
			heading = self.data.heading_of(node, self.t)
			mag = np.linalg.norm(np.asarray(heading))
			colorval = self.plot_head_mag_norm(mag)
			if self.frame.args.verbose: self.log.debug(
				"Average heading: %s, Color: [%s], Speed: %s" % (str(heading), str(colorval), str(mag)))

			xs, ys, zs = zip(position, np.add(position, (np.asarray(heading) * 50)))
			self.node_vector_collections[node] = Arrow3D(
				xs, ys, zs,
				mutation_scale = 2, lw = 1,
				arrowstyle = "-|>", color = self.plot_sphere_cm(colorval), alpha = self.vector_opacity
			)
			self.plot_axes.add_artist(
				self.node_vector_collections[node],
			)

	def _remove_sphere(self):
		if isinstance(self.sphere_line_collection, Line3DCollection)\
		and self.sphere_line_collection in self.plot_axes.collections:
			self.plot_axes.collections.remove(self.sphere_line_collection)

	def _remove_vectors(self):
		if self.node_vector_collections is not None:
			for arrow in self.node_vector_collections:
				if arrow is not None:
					arrow.remove()
		self.node_vector_collections = [None for i in range(self.data.n)]

	def update_metric_charts(self):
		self.ctl.update_metrics()

		for i, plot in enumerate(self.metrics.ctl):
			self.metric_axes[i] = plot.plot(wanted = np.asarray(self.displayed_nodes))
			self.metric_xlines[i] = self.metric_axes[i].axvline(x = self.t, color = 'r', linestyle = ':')
			self.metric_axes[i].relim()
			xlim = (max(0, self.t - 100), max(100, self.t + 100))
			self.metric_axes[i].set_xlim(xlim)
			self.metric_axes[i].set_ylim(*plot.ylim(xlim))
		self.metrics[-1].ax.get_xaxis().set_visible(True)

	####
	# Button Event Handlers
	####
	def on_pause_btn(self, event):
		self.paused = not self.paused

	def on_update_pause_btn(self, event):
		self.pause_btn.SetLabel("Resume" if self.paused else "Pause")

	def on_play_btn(self, event):
		self.redraw_plot(t = 0)
		self.paused = False

	def on_faster_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 1.1), self.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)

	def on_slower_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 0.9), self.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)

	def on_time_slider(self, event):
		t = self.time_slider.GetValue()
		self.log.debug("Slider: Setting time to %d" % t)
		wx.CallAfter(self.redraw_plot, t = t)

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
		self.trail = int(norm_trail * (self.data.tmax / 100.0))
		self.log.debug("Slider: Setting trail to %d" % self.trail)
		wx.CallAfter(self.redraw_plot)

	####
	# Menu Event Handlers
	####
	def on_node_select(self, event):
		"""
		Based on the wxPython demo - opens the MultiChoiceDialog
		and sets that selection as the node entries to be
		displayed
		"""
		lst = list(self.data.names)
		dlg = wx.MultiChoiceDialog(self,
		                           "Select nodes",
		                           "wx.MultiChoiceDialog", lst)

		selections = [i for i in range(self.data.n) if self.displayed_nodes[i]]
		print selections
		dlg.SetSelections(selections)

		if (dlg.ShowModal() == wx.ID_OK):
			selections = dlg.GetSelections()
			strings = [lst[x] for x in selections]
			print "You chose:" + str(strings)
			self.displayed_nodes.fill(False)
			for checked in selections:
				self.displayed_nodes[checked] = True

			self.plot_analysis()

		wx.CallAfter(self.redraw_plot)

	def on_idle(self, event):
		if not self.paused:
			self.move_T()

	def on_resize(self, event):
		self.resize_plots()