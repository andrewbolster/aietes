__author__ = 'andrewbolster'

import wx
from wx.lib.agw.pycollapsiblepane import PyCollapsiblePane as PCP

import os
import logging
import traceback, sys

import matplotlib

matplotlib.use('WXAgg')

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib import cm

matplotlib.rcParams.update({'font.size': 8})

import numpy as np

WIDTH, HEIGHT = 8, 6
SIDEBAR_WIDTH = 2

class MetricView():
	'''
	This Class is a plotable view of the Metric class availiable from Bounos.
	It is instantiated with the representative Bounos.Metric base metric
	'''

	def __init__(self, axes, base_metric, *args, **kw):
		self.ax = axes
		self.data = base_metric.data.view()
		self.label = base_metric.label
		self.highlight_data = base_metric.highlight_data
		self.ndim = 0
		if __debug__: logging.debug("%s" % self)

	def plot(self, wanted = None, time = None):
		"""
		Update the Plot based on 'new' wanted data
		"""
		self.ax.clear()
		self.ax.set_ylabel(self.label)
		self.ax.get_xaxis().set_visible(True)

		if all(wanted == True) or self.ndim == 1:
			self.ax.plot(self.data, alpha = 0.3)
		else:
			logging.info("Printing %s with Wanted:%s" % (self, wanted))
			self.ax.plot(np.ndarray(buffer = self.data, shape = self.data.shape)[:, wanted], alpha = 0.3)

		if self.highlight_data is not None:
			self.ax.plot(self.highlight_data, color = 'k', linestyle = '--')
		return self.ax

	def ylim(self, xlim, margin = None):
		(xmin, xmax) = xlim
		if self.highlight_data is not None:
			data = np.append(self.data, self.highlight_data).reshape((self.data.shape[0], -1))
		else:
			data = self.data
		if self.ndim > 1:
			slice = data[xmin:xmax][:]
		else:
			slice = data[xmin:xmax]
		try:
			ymin = slice.min()
			ymax = slice.max()
			range = ymax - ymin
			if margin is None:
				margin = range * 0.2

			ymin -= margin
			ymax += margin
		except ValueError as e:
			raise e
		return (ymin, ymax)


class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
		FancyArrowPatch.draw(self, renderer)


class GenericFrame(wx.Frame):
	def __init__(self, controller, *args, **kw):
		wx.Frame.__init__(self, None, title = EphyraNotebook.description, *args, **kw)
		self.log = logging.getLogger(self.__module__)
		self.ctl = controller
		p = VisualNavigator(self, self, wx.ID_ANY)
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(p, proportion = 1, flag = wx.GROW)
		self.SetMinSize((800, 600))
		self.SetSizer(sizer)
		sizer.Fit(p)
		self.Layout()


class EphyraNotebook(wx.Frame):
	def __init__(self, controller, *args, **kw):
		wx.Frame.__init__(self, None, title = "Ephyra")
		self.log = logging.getLogger(self.__module__)
		self.ctl = controller
		self.args = kw.get("exec_args", None)

		# Create a panel and a Notebook on the Panel
		self.p = wx.Panel(self)
		self.nb = wx.Notebook(self.p)
		self.CreateMenuBar()

		self.Bind(wx.EVT_SIZE, self.on_resize)
		self.Bind(wx.EVT_IDLE, self.on_idle)

		pages = [VisualNavigator, Configurator, Simulator]

		for page in pages:
			self.log.debug("Adding Page: %s" % page.__name__)
			self.nb.AddPage(page(self.nb, self), page.__name__)

		self.status_bar = self.CreateStatusBar()
		self.status_bar.SetFieldsCount(3)

		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.sizer.Add(self.nb, proportion = 1, flag = wx.GROW | wx.ALL)
		self.p.SetSizer(self.sizer)
		self.SetMinSize((800, 600))
		self.Layout()
		self.Show()


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
			wx.CallAfter(self.exit)

	def exit(self):
		self.DestroyChildren()
		self.Destroy()

	def on_idle(self, event):
		self.nb.GetCurrentPage().on_idle(event)


	def on_resize(self, event):
		plot_size = self.GetClientSize()
		self.log.debug("plotsize:%s" % str(plot_size))

		self.p.SetSize(plot_size)

		try:
			self.nb.GetCurrentPage().on_resize(event)
		except:
			traceback.print_exc(file = sys.stdout)
			wx.CallAfter(self.exit)

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

	def update_timing(self, t, tmax):
		self.status_bar.SetStatusText("%s:%d/%d" % ("SIM" if self.ctl.is_simulation() else "REC", t, tmax), number = 2)

#####################################################################################################
##
#####################################################################################################
class Configurator(wx.Panel):
	"""
	The Configurator panel allows the user to generate aietes-compatible configurations

	The general operation is two fold;
		*Editing simulation environment
		*Editing simulation defaults for Nodes
		*Editing simulation defaults for Behaviours
		*Editing simulation defaults for Applications
		*Individually editing modifications to Nodes, Applications, and Behaviours

	The editable Characteristics of Nodes in this are:
		*Application selection
		*Behaviour Selection
		*Speeds (Cruising, Max)
		*Max Turn rate
		*Initial Position

	The editible Characteristics of Behaviours in this are:
		*Factors (Clumping, Repulsion, Schooling, Waypointing)
		*Distances (Collision Avoidance, Min/Max Neighbourhood)
		*Nearest Neighbour Count
		*Update rate

	The editable Characteristics of Applications in this are:
		*NONE #ToDo

	"""

	def __init__(self, parent, frame, *args, **kw):
		wx.Panel.__init__(self, parent, *args, **kw)
		self.v_sizer = wx.BoxSizer(wx.VERTICAL)
		self.v_sizer.Add(NodeConfigurator(self, title = "Node Configuration"), flag = wx.EXPAND | wx.ALIGN_LEFT)
		self.SetSizer(self.v_sizer, wx.EXPAND)
		self.Layout()

	def on_resize(self, event):
		self.Layout()



	def on_idle(self, event):
		pass


##########################
## Configurator Helpers
##########################
class NodeConfigurator(wx.Panel):
	"""
	This class provides a vertically sized layout of PyCollapsiblePanels to configure Nodes with
	"""

	def __init__(self, parent, *args, **kw):
		self.cp_style = wx.CP_DEFAULT_STYLE | wx.CP_NO_TLW_RESIZE
		title = kw.pop("title", "BOOBIES")
		wx.Panel.__init__(self, parent, *args, **kw)
		self.content_sizer = wx.BoxSizer(wx.VERTICAL)
		self.cp = cp = PCP(self, label = title, style = self.cp_style)
		self.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, self.on_toggle, cp)
		self._custom_position = False
		self.make_content(cp.GetPane())

	def on_toggle(self, event):
		self.cp.Collapse(self.cp.IsExpanded())

	def make_content(self, pane):
		"""
		Generate the internals of the panel
			* Initial Position - Custom/random/whatever
			* Velocity Control - Cruising, Speed, Turn
			* Behaviour - Another Configurator
		"""
		self.sizer = wx.BoxSizer(wx.VERTICAL)

		pos_lbl = wx.StaticText(pane, label = "Initial Position")
		pos_list = ["Custom", "Random", "Centre"]
		pos_btn = wx.RadioBox(pane, wx.ID_ANY, "Position", (10, 10), wx.DefaultSize, pos_list, 1, wx.RA_SPECIFY_COLS)

		self.pos_custom_x = wx.TextCtrl(pane, wx.ID_ANY, "500", (20, 20))
		self.pos_custom_y = wx.TextCtrl(pane, wx.ID_ANY, "500", (20, 20))
		self.pos_custom_z = wx.TextCtrl(pane, wx.ID_ANY, "500", (20, 20))
		pos_custom_sizer = wx.BoxSizer(wx.HORIZONTAL)
		for pos in [self.pos_custom_x, self.pos_custom_y, self.pos_custom_z]:
			pos.Enable(False)
			pos_custom_sizer.Add(pos)

		pos_sizer = wx.BoxSizer(wx.HORIZONTAL)
		pos_sizer.Add(pos_btn)
		pos_sizer.Add(pos_custom_sizer)

		vel_lbl = wx.StaticText(pane, label = "Velocity Control")

		beh_lbl = wx.StaticText(pane, label = "Behaviour Control")

		self.sizer.Add(pos_lbl, wx.EXPAND)
		self.sizer.Add(pos_sizer, wx.EXPAND)
		self.sizer.Add(vel_lbl, wx.EXPAND)
		self.sizer.Add(beh_lbl, wx.EXPAND)

		pane.SetSizerAndFit(self.sizer)


class Simulator(wx.Panel):
	def __init__(self, parent, frame, *args, **kw):
		wx.Panel.__init__(self, parent, *args, **kw)
		wx.StaticText(self, wx.ID_ANY, "This is the Simulator", (20, 20))


	def on_idle(self, event):
		pass


class VisualNavigator(wx.Panel):
	def __init__(self, parent, frame, *args, **kw):
		wx.Panel.__init__(self, parent, *args, **kw)
		self.log = frame.log.getChild(self.__class__.__name__)
		self.frame = frame
		self.ctl = frame.ctl

		# Timing and playback defaults
		self.paused = None
		self.t = 0
		self.tmax = None # Not always the same as the data tmax eg simulating
		self.d_t = 1

		# Configure Plot Display
		self.trail_opacity = 0.7
		self.trail_length = 100

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

		# Configure Plotting Panel (plot_pnl)
		self._plot_initialised = False
		self.plot_pnl = wx.Panel(self)
		self.fig = Figure()
		self.canvas = FigureCanvas(self.plot_pnl, -1, self.fig)
		self.axes = self.fig.add_axes([0, 0, 1, 1], )
		self.gs = GridSpec(HEIGHT, WIDTH) # (height,width)

		####
		# Main Plot
		####
		plot_area = self.gs[:-1, SIDEBAR_WIDTH:]
		self.plot_axes = self.fig.add_subplot(plot_area, projection = '3d')
		self.lines = []

		####
		# Metrics
		####
		metric_areas = [self.gs[x, :SIDEBAR_WIDTH] for x in range(HEIGHT)]
		self.metric_axes = [self.fig.add_subplot(metric_areas[i]) for i in range(HEIGHT)]
		for ax in self.metric_axes:
			ax.autoscale_view(scalex = False, tight = True)
		self.metric_xlines = [None for i in range(HEIGHT)]
		self.metric_views = [None for i in range(HEIGHT)]

		# Configure Control Panel
		self.control_pnl = wx.Panel(self)
		self.time_slider = wx.Slider(self.control_pnl, value = 0, minValue = 0, maxValue = 1)
		self.pause_btn = wx.Button(self.control_pnl, label = "Pause")
		self.play_btn = wx.Button(self.control_pnl, label = "Play")
		self.faster_btn = wx.Button(self.control_pnl, label = "Rate++")
		self.slower_btn = wx.Button(self.control_pnl, label = "Rate--")
		self.trail_slider = wx.Slider(self.control_pnl, value = self.trail_length, minValue = 0, maxValue = 100,
		                              size = (120, -1))

		self.Bind(wx.EVT_SCROLL, self.on_time_slider, self.time_slider)
		self.Bind(wx.EVT_SCROLL, self.on_trail_slider, self.trail_slider)
		self.Bind(wx.EVT_BUTTON, self.on_pause_btn, self.pause_btn)
		self.Bind(wx.EVT_BUTTON, self.on_play_btn, self.play_btn)
		self.Bind(wx.EVT_BUTTON, self.on_faster_btn, self.faster_btn)
		self.Bind(wx.EVT_BUTTON, self.on_slower_btn, self.slower_btn)

		control_sizer = wx.BoxSizer(wx.VERTICAL)
		time_sizer = wx.BoxSizer(wx.HORIZONTAL)
		control_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

		time_sizer.Add(self.time_slider, proportion = 1)
		control_btn_sizer.Add(self.pause_btn)
		control_btn_sizer.Add(self.play_btn, flag = wx.RIGHT, border = 5)
		control_btn_sizer.Add(self.faster_btn, flag = wx.LEFT, border = 5)
		control_btn_sizer.Add(self.slower_btn)
		control_btn_sizer.Add(self.trail_slider, flag = wx.TOP | wx.LEFT, border = 5)

		#Metric Buttons
		self.sphere_chk = wx.CheckBox(self.control_pnl, label = "Sphere")
		self.sphere_chk.SetValue(self.sphere_enabled)
		self.Bind(wx.EVT_CHECKBOX, self.on_sphere_chk, self.sphere_chk)
		control_btn_sizer.Add(self.sphere_chk)

		self.vector_chk = wx.CheckBox(self.control_pnl, label = "Vector")
		self.vector_chk.SetValue(self.node_vector_enabled)
		self.Bind(wx.EVT_CHECKBOX, self.on_vector_chk, self.vector_chk)
		control_btn_sizer.Add(self.vector_chk)

		control_sizer.Add(time_sizer, flag = wx.EXPAND | wx.BOTTOM, border = 10)
		control_sizer.Add(control_btn_sizer, proportion = 1, flag = wx.EXPAND)
		self.control_pnl.SetSizer(control_sizer)

		self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
		self.panel_sizer.Add(self.plot_pnl, proportion = 1, flag = wx.EXPAND | wx.ALL)
		self.panel_sizer.Add(self.control_pnl, flag = wx.EXPAND | wx.BOTTOM)

		self.SetSizer(self.panel_sizer)

		if self.ctl.model_is_ready():
			self.log.debug("Model is ready, initialising 3D plot")
			self.initialise_3d_plot()
		else:
			self.log.debug("Model not ready yet")

	####
	# 3D Plot Handling Functions
	####
	def initialise_3d_plot(self):
		self._plot_initialised = True
		self.log.info("Plot Initialised")
		# Start off with all nodes displayed
		self.displayed_nodes = np.empty(self.ctl.get_n_vectors(), dtype = bool)
		self.displayed_nodes.fill(True)

		# Initialise Sphere data anyway
		self.plot_sphere_cm = cm.Spectral_r

		# Initialise Vector display data anyway
		(hmax, hmin) = self.ctl.get_heading_mag_max_min()
		(pmax, pmin) = self.ctl.get_position_stddev_max_min()
		self.plot_head_mag_norm = Normalize(vmin = hmin, vmax = hmax)
		self.plot_pos_stddev_norm = Normalize(vmin = pmin, vmax = pmax)

		# Initialse Positional Plot
		shape = self.ctl.get_extent()
		self.log.info(shape)
		self.plot_axes.set_title("Tracking overview of %s" % self.ctl.get_model_title())
		self.plot_axes.set_xlim3d((0, shape[0]))
		self.plot_axes.set_ylim3d((0, shape[1]))
		self.plot_axes.set_zlim3d((0, shape[2]))
		self.plot_axes.set_xlabel('X')
		self.plot_axes.set_ylabel('Y')
		self.plot_axes.set_zlabel('Z')

		# Initialise 3D Plot Lines
		(xs, ys, zs) = self.ctl.get_3D_trail()
		self.log.info("Got %s" % str(xs.shape))

		self.lines = [self.plot_axes.plot(x, y, z, alpha = self.trail_opacity)[0] for x, y, z in zip(xs, ys, zs)]

		# Initialise Metric Views
		metrics = self.ctl.get_metrics()
		assert len(metrics) == HEIGHT, str(metrics)
		for i, (axes, metric) in enumerate(zip(self.metric_axes, self.ctl.get_metrics())):
			self.metric_views[i] = MetricView(axes, metric)

		#Initialise Data-Based timings.
		self.tmax = max(self.ctl.get_final_tmax() - 1, 1)
		self.time_slider.SetMax(self.tmax)
		self.time_slider.SetValue(0)
		self.d_t = int((self.tmax + 1) / 100)

	def redraw_page(self, t = None):
		if not self._plot_initialised:
			raise RuntimeError("Plot isn't ready yet!")
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
			(xs, ys, zs) = self.ctl.get_3D_trail(node = n, time_start = self.t, length = self.trail_length)

			line.set_data(xs, ys)
			line.set_3d_properties(zs)
			line.set_label(self.ctl.get_vector_names(i = n))

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
		if self.ctl.model_is_ready():
			tmax = self.ctl.get_final_tmax()
		else:
			tmax = None

		# If trying to go over the end, don't, and either stop or loop
		if tmax is not None and  t > tmax:
			if self.frame.args.loop:
				self.log.debug("Looping")
				t = 0
			else:
				self.log.debug("End Of The Line")
				self.paused = True
				t = tmax
				if self.frame.args.autoexit:
					wx.CallAfter(self.frame.exit)
					self.Destroy()

		if t < 0:
			self.log.debug("Tried to reverse, pausing")
			self.paused = True
			t = 0

		self.frame.update_timing(t, self.tmax)
		self.time_slider.SetValue(t)
		if self.ctl.model_is_ready():
			self.redraw_page(t = t)


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
		fleet = self.ctl.get_fleet_configuration(self.t)
		(x, y, z) = fleet['positions']['avg']
		(r, s) = (max(fleet['positions']['delta_avg']), fleet['positions']['stddev'])

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
		positions = self.ctl.get_fleet_positions(self.t)
		headings = self.ctl.get_fleet_headings(self.t)

		for node in range(self.ctl.get_n_vectors()):
			mag = np.linalg.norm(np.asarray(headings[node]))
			colorval = self.plot_head_mag_norm(mag)
			if self.frame.args.verbose: self.log.debug(
				"Average heading: %s, Color: [%s], Speed: %s" % (str(headings[node]), str(colorval), str(mag)))

			xs, ys, zs = zip(positions[node], np.add(positions[node], (np.asarray(headings[node]) * 50)))
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
		self.node_vector_collections = [None for i in range(self.ctl.get_n_vectors())]

	def update_metric_charts(self):
		for i, plot in enumerate(self.metric_views):
			self.metric_axes[i] = plot.plot(wanted = np.asarray(self.displayed_nodes))
			self.metric_xlines[i] = self.metric_axes[i].axvline(x = self.t, color = 'r', linestyle = ':')
			self.metric_axes[i].relim()
			xlim = (max(0, self.t - 100), max(100, self.t + 100))
			self.metric_axes[i].set_xlim(xlim)
			self.metric_axes[i].set_ylim(*plot.ylim(xlim))
		self.metric_views[-1].ax.get_xaxis().set_visible(True)

	####
	# Button Event Handlers
	####
	def on_pause_btn(self, event):
		self.paused = not self.paused
		self.pause_btn.SetLabel("Resume" if self.paused else "Pause")

	def on_play_btn(self, event):
		self.paused = False
		wx.CallAfter(self.redraw_page, t = 0)

	def on_faster_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 1.1), self.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)

	def on_slower_btn(self, event):
		self.d_t = int(min(max(1, self.d_t * 0.9), self.tmax / 2))
		self.log.debug("Setting time step to: %s" % self.d_t)

	def on_time_slider(self, event):
		t = self.time_slider.GetValue()
		self.log.debug("Slider: Setting time to %d" % t)
		wx.CallAfter(self.redraw_page, t = t)

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
		wx.CallAfter(self.redraw_page)

	def on_vector_chk(self, event):
		if not self.vector_chk.IsChecked():
			self._remove_vectors()
			self.log.debug("Vector Overlay Disabled")
			self.node_vector_enabled = False
		else:
			self.log.debug("Vector Overlay Enabled")
			self.node_vector_enabled = True
		wx.CallAfter(self.redraw_page)

	def on_trail_slider(self, event):
		event.Skip()
		norm_trail = self.trail_slider.GetValue()
		self.trail_length = int(norm_trail * (self.ctl.get_final_tmax() / 100.0))
		self.log.debug("Slider: Setting trail to %d" % self.trail_length)
		wx.CallAfter(self.redraw_page)

	####
	# Menu Event Handlers
	####
	def on_node_select(self, event):
		"""
		Based on the wxPython demo - opens the MultiChoiceDialog
		and sets that selection as the node entries to be
		displayed
		"""
		lst = list(self.ctl.get_vector_names())
		dlg = wx.MultiChoiceDialog(self,
		                           "Select nodes",
		                           "wx.MultiChoiceDialog", lst)

		selections = [i for i in range(self.ctl.get_n_vectors()) if self.displayed_nodes[i]]
		print selections
		dlg.SetSelections(selections)

		if (dlg.ShowModal() == wx.ID_OK):
			selections = dlg.GetSelections()
			strings = [lst[x] for x in selections]
			print "You chose:" + str(strings)
			self.displayed_nodes.fill(False)
			for checked in selections:
				self.displayed_nodes[checked] = True

			self.update_metric_charts()

		wx.CallAfter(self.redraw_page)

	def on_idle(self, event):
		if not self.paused:
			if self._plot_initialised:
				self.move_T()
			elif self.ctl.model_is_ready():
				self.initialise_3d_plot()
			else:
				pass

	def on_resize(self, event):
		plot_size = self.panel_sizer.GetChildren()[0].GetSize()
		self.log.debug("plotsize:%s" % str(plot_size))

		self.plot_pnl.SetSize(plot_size)
		self.canvas.SetSize(plot_size)
		self.fig.set_size_inches(float(plot_size[0]) / self.fig.get_dpi(),
		                         float(plot_size[0]) / self.fig.get_dpi()
		)
