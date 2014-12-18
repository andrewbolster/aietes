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

WIDTH, HEIGHT = 16, 9
SIDEBAR_WIDTH = 8

import logging

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib import cm

from ephyra import wx
from ephyra.Views import MetricView, Arrow3D
from aietes.Tools import timeit, mag



# noinspection PyStringFormat
class VisualNavigator(wx.Panel):
    @timeit()
    def __init__(self, parent, frame, *args, **kw):
        wx.Panel.__init__(self, parent, *args, **kw)
        self.log = frame.log.getChild(self.__class__.__name__)
        self.frame = frame
        self.ctl = frame.ctl

        # Timing and playback defaults
        self.paused = None
        self.t = 0
        self.tmax = None  # Not always the same as the data tmax eg simulating
        self.d_t = 1

        # Configure Plot Display
        self.trail_opacity = 0.7
        self.trail_length = 100

        # Configure Sphere plotting on plot_pnl
        self.sphere_enabled = True
        self.sphere_line_collection = None
        self.sphere_opacity = 0.9

        # Configure Vector Plotting on Plot_pnl
        self.node_vector_enabled = True
        self.fleet_vector_enabled = True
        self.node_vector_collections = []
        self.fleet_vector_collection = None
        self._fleet_zoom = False
        self.vector_opacity = 0.9

        # Configure contrib Plotting on Plot_pnl
        self.node_contrib_enabled = False
        self.fleet_contrib_enabled = False
        self.node_contrib_collections = []
        self.fleet_contrib_collection = None
        self.contrib_opacity = 0.9

        # Configure Plotting Panel (plot_pnl)
        self._plot_initialised = False
        self.plot_pnl = wx.Panel(self)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.plot_pnl, -1, self.fig)
        self.axes = self.fig.add_axes([0, 0, 1, 1], )
        self.gs = GridSpec(HEIGHT, WIDTH)  # (height,width)

        ####
        # Main Plot
        ####
        plot_area = self.gs[:-1, SIDEBAR_WIDTH:]
        self.plot_axes = self.fig.add_subplot(
            plot_area, projection='3d', aspect=1)
        self.lines = []

        ####
        # Metrics
        ####
        metric_areas = [self.gs[x, :SIDEBAR_WIDTH] for x in range(HEIGHT)]
        self.metric_axes = [
            self.fig.add_subplot(metric_areas[i]) for i in range(HEIGHT)]
        for ax in self.metric_axes:
            ax.autoscale_view(scalex=False, tight=True)
        self.metric_xlines = [None for i in range(HEIGHT)]
        self.achievement_xlines = [None for i in range(HEIGHT)]
        self.metric_views = [None for i in range(HEIGHT)]
        self._zoom_metrics = True

        # Configure Control Panel
        self.control_pnl = wx.Panel(self)
        self.time_slider = wx.Slider(
            self.control_pnl, value=0, minValue=0, maxValue=1)
        self.pause_btn = wx.Button(self.control_pnl, label="Pause")
        self.play_btn = wx.Button(self.control_pnl, label="Play")
        self.faster_btn = wx.Button(self.control_pnl, label="Rate++")
        self.slower_btn = wx.Button(self.control_pnl, label="Rate--")
        self.trail_slider = wx.Slider(self.control_pnl, value=self.trail_length, minValue=0, maxValue=100,
                                      size=(120, -1))

        self.Bind(wx.EVT_SCROLL, self.on_time_slider, self.time_slider)
        self.Bind(wx.EVT_SCROLL, self.on_trail_slider, self.trail_slider)
        self.Bind(wx.EVT_BUTTON, self.on_pause_btn, self.pause_btn)
        self.Bind(wx.EVT_BUTTON, self.on_play_btn, self.play_btn)
        self.Bind(wx.EVT_BUTTON, self.on_faster_btn, self.faster_btn)
        self.Bind(wx.EVT_BUTTON, self.on_slower_btn, self.slower_btn)

        control_sizer = wx.BoxSizer(wx.VERTICAL)
        time_sizer = wx.BoxSizer(wx.HORIZONTAL)
        control_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        time_sizer.Add(self.time_slider, proportion=1)
        control_btn_sizer.Add(self.pause_btn)
        control_btn_sizer.Add(self.play_btn, flag=wx.RIGHT, border=5)
        control_btn_sizer.Add(self.faster_btn, flag=wx.LEFT, border=5)
        control_btn_sizer.Add(self.slower_btn)
        control_btn_sizer.Add(
            self.trail_slider, flag=wx.TOP | wx.LEFT, border=5)

        # Metric Buttons
        self.sphere_chk = wx.CheckBox(self.control_pnl, label="Sphere")
        self.sphere_chk.SetValue(self.sphere_enabled)
        self.Bind(wx.EVT_CHECKBOX, self.on_sphere_chk, self.sphere_chk)
        control_btn_sizer.Add(self.sphere_chk)

        self.metric_zoom_chk = wx.CheckBox(
            self.control_pnl, label="Metric Zoom")
        self.metric_zoom_chk.SetValue(self._zoom_metrics)
        self.Bind(
            wx.EVT_CHECKBOX, self.on_metric_zoom_chk, self.metric_zoom_chk)
        control_btn_sizer.Add(self.metric_zoom_chk)

        self.zoom_fleet_chk = wx.CheckBox(self.control_pnl, label="Fleet Zoom")
        self.zoom_fleet_chk.SetValue(self._fleet_zoom)
        self.Bind(wx.EVT_CHECKBOX, self.on_zoom_fleet_chk, self.zoom_fleet_chk)
        control_btn_sizer.Add(self.zoom_fleet_chk)

        self.vector_chk = wx.CheckBox(self.control_pnl, label="Vector")
        self.vector_chk.SetValue(self.node_vector_enabled)
        self.Bind(wx.EVT_CHECKBOX, self.on_vector_chk, self.vector_chk)
        control_btn_sizer.Add(self.vector_chk)

        self.contrib_chk = wx.CheckBox(self.control_pnl, label="Contribs.")
        self.contrib_chk.SetValue(self.node_contrib_enabled)
        self.Bind(wx.EVT_CHECKBOX, self.on_contrib_chk, self.contrib_chk)
        control_btn_sizer.Add(self.contrib_chk)

        control_sizer.Add(time_sizer, flag=wx.EXPAND | wx.BOTTOM, border=10)
        control_sizer.Add(control_btn_sizer, proportion=1, flag=wx.EXPAND)
        self.control_pnl.SetSizer(control_sizer)

        self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel_sizer.Add(
            self.plot_pnl, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.panel_sizer.Add(self.control_pnl, flag=wx.EXPAND | wx.BOTTOM)

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
        """


        """
        self._plot_initialised = True
        self.log.info("Plot Initialised")
        # Start off with all nodes displayed
        self.displayed_nodes = np.empty(self.ctl.get_n_vectors(), dtype=bool)
        self.displayed_nodes.fill(True)

        # Initialise Sphere data anyway
        self.plot_sphere_cm = cm.Spectral_r

        # Initialise Vector display data anyway
        (hmax, hmin) = self.ctl.get_heading_mag_max_min()
        (pmax, pmin) = self.ctl.get_position_stddev_max_min()
        self.plot_head_mag_norm = Normalize(vmin=hmin, vmax=hmax)
        self.plot_pos_stddev_norm = Normalize(vmin=pmin, vmax=pmax)

        # Initialse Positional Plot
        shape = self.ctl.get_extent()
        self.log.info(shape)
        self.plot_axes.set_title(
            "Tracking overview of %s" % self.ctl.get_model_title())
        self.plot_axes.set_xlim3d((0, shape[0]))
        self.plot_axes.set_ylim3d((0, shape[1]))
        self.plot_axes.set_zlim3d((0, shape[2]))
        self.plot_axes.set_xlabel('X')
        self.plot_axes.set_ylabel('Y')
        self.plot_axes.set_zlabel('Z')

        # Initialise 3D Plot Lines
        (xs, ys, zs) = self.ctl.get_3D_trail()
        self.names = self.ctl.get_vector_names()
        self.log.info("Got %s" % str(xs.shape))

        self.lines = [
            self.plot_axes.plot(x, y, z, label=self.names[i], alpha=self.trail_opacity)[0] for
            i, (x, y, z) in enumerate(zip(xs, ys, zs))]
        self._legend = self.plot_axes.legend(loc="lower right")

        self.labels = [
            self.plot_axes.text(x[0], y[0], z[0], self.names[i][0])
            for i, (x, y, z) in enumerate(zip(xs, ys, zs))
        ]

        # Initialise Metric Views
        metrics = self.ctl.get_metrics()
        if len(metrics) != len(self.metric_axes):
            self.log.info(
                "Not enough height for selected metrics (%s) " % len(metrics))
        for i, (axes, metric) in enumerate(zip(self.metric_axes, metrics)):
            self.metric_views[i] = MetricView(axes, metric)

        # Initialise Waypoints if present
        waypoints = self.ctl.get_waypoints()
        if waypoints is not None and waypoints.size > 0:
            self.log.debug("Found [%s] waypoints" %
                           str(getattr(waypoints, "shape", "No Shape")))
            # Case where there is a single common waypoint set, set up spheres
            if waypoints.ndim == 2:
                for (x, y, z), r in waypoints:
                    xs, ys, zs = self.sphere(x, y, z, r)
                    self.plot_axes.plot_wireframe(xs, ys, zs, alpha=0.1)
            else:
                # If there are multiple, sphere, colour, AND lines matching
                # each index
                for i, inner_waypoints in enumerate(waypoints):
                    xs, ys, zs = [], [], []
                    for (x, y, z), r in inner_waypoints:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                    self.plot_axes.plot(
                        xs, ys, zs, alpha=0.1, color=self.lines[i].get_color())
        else:
            self.log.debug("No Waypoints Defined")

        # Initialise Data-Based timings.
        self.tmax = max(self.ctl.get_final_tmax() - 1, 1)
        self.time_slider.SetMax(self.tmax)
        self.time_slider.SetValue(0)
        self.d_t = int((self.tmax + 1) / 100)

    def redraw_page(self, t=None):
        """

        :param t:
        :return: :raise:
        """
        if not self._plot_initialised:
            self.log.warn("Plot isn't ready yet!")
            return
            ###
        # Update Time!
        ###
        if t is not None:
            self.t = t
            del t

        ###
        # MAIN PLOT AREA
        ###
        for n, (line, label) in enumerate(zip(self.lines, self.labels)):
            (xs, ys, zs) = self.ctl.get_3D_trail(
                node=n, time_start=self.t, length=self.trail_length)
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            line.set_label(self.names[n])
            try:
                label.set_position((xs[0], ys[0]))
                label.set_3d_properties(zs[0])
            except TypeError as err:
                self.log.error("x:%s,y:%s,z:%s" % (xs[0], ys[0], zs[0]))
                raise
            except IndexError:
                # In the case of the 'first time'. This should probably be removed in the case
                # of any architectural changes.
                pass

        ###
        # VECTOR OVERLAYS TO MAIN PLOT AREA
        ###
        # Arrow3D
        if self.node_vector_enabled:
            self.redraw_fleet_heading_vectors()
        elif self.node_contrib_enabled:
            self.redraw_fleet_heading_contribs()
            ###
        # SPHERE OVERLAY TO MAIN PLOT AREA
        ###
        if self.sphere_enabled:
            self.redraw_fleet_sphere()

        self.update_metric_charts()

        ###
        # UPDATE WINDOW PARAMETERS
        ###
        if self.zoom_fleet_chk.IsChecked():
            # On first time, zoom to fleet sphere, otherwise use whatever the
            # width of the window is.
            if not self._fleet_zoom:
                self._fleet_zoom = True
                (lx, rx) = self.plot_axes.get_xlim3d()
                (ly, ry) = self.plot_axes.get_ylim3d()
                (lz, rz) = self.plot_axes.get_zlim3d()
            else:
                (lx, rx), (ly, ry), (lz, rz) = self.ctl.get_position_min_max(
                    self.t)
            x_width = abs(lx - rx)
            y_width = abs(ly - ry)
            z_width = abs(lz - rz)
            width = max(x_width, y_width, z_width) * 1.1

            positions = self.ctl.get_fleet_positions(self.t)
            avg = np.average(positions, axis=0)
            self.plot_axes.set_xlim3d(
                (avg[0] - (width / 2), avg[0] + (width / 2)))
            self.plot_axes.set_ylim3d(
                (avg[1] - (width / 2), avg[1] + (width / 2)))
            self.plot_axes.set_zlim3d(
                (avg[2] - (width / 2), avg[2] + (width / 2)))

        if self._legend not in self.plot_axes.get_children():
            self.plot_axes.add_artist(self._legend)

        wx.CallAfter(self.canvas.draw)

    def move_T(self, delta_t=None):
        """ Seek the visual plot by delta_t while doing bounds checking and redraw

        :param delta_t:
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
        if tmax is not None and t > tmax:
            if self.frame.args.loop:
                self.log.debug("Looping")
                t = 0
            else:
                self.log.debug("End Of The Line")
                self.paused = True
                t = tmax
                if self.frame.args.autoexit and self.frame.args.autostart:
                    wx.CallAfter(self.frame.exit)
                    self.Destroy()

        if t < 0:
            self.log.debug("Tried to reverse, pausing")
            self.paused = True
            t = 0

        self.frame.update_timing(t, self.tmax)
        self.time_slider.SetValue(t)
        if self.ctl.model_is_ready():
            self.redraw_page(t=t)

    @staticmethod
    def sphere(x, y, z, r=1.0):
        """
        Returns a sphere definition tuple (xs,ys,zs) for use with plot_wireframe
        :param x:
        :param y:
        :param z:
        :param r:
        """
        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        xs = (r * np.cos(u) * np.sin(v)) + x
        ys = (r * np.sin(u) * np.sin(v)) + y
        zs = (r * np.cos(v)) + z

        return xs, ys, zs

    def redraw_fleet_sphere(self):
        """


        """
        fleet = self.ctl.get_fleet_configuration(self.t)
        (x, y, z) = fleet['positions']['avg']
        (r, s) = (max(fleet['positions']['delta_avg']),
                  fleet['positions']['stddev'])

        xs, ys, zs = self.sphere(x, y, z, r)
        colorval = self.plot_pos_stddev_norm(s)
        if self.frame.args.verbose:
            self.log.debug("Average position: %s, Color: %s[%s], StdDev: %s" % (
                str((x, y, z)), str(self.plot_sphere_cm(colorval)), str(colorval), str(s)))

        self._remove_sphere()

        # TODO Update to UPDATE DATA instead of re plotting
        self.sphere_line_collection = self.plot_axes.plot_wireframe(xs, ys, zs,
                                                                    alpha=self.sphere_opacity,
                                                                    color=self.plot_sphere_cm(
                                                                        colorval)
        )

    def redraw_fleet_heading_vectors(self):
        """


        """
        self._remove_vectors(self.node_vector_collections)
        positions = self.ctl.get_fleet_positions(self.t)
        headings = self.ctl.get_fleet_headings(self.t)

        for node in range(self.ctl.get_n_vectors()):
            magnitude = mag(np.asarray(headings[node]))
            colorval = self.plot_head_mag_norm(magnitude)
            if self.frame.args.verbose:
                self.log.debug(
                    "Average heading: %s, Color: [%s], Speed: %s" % (str(headings[node]), str(colorval), str(magnitude)))

            xs, ys, zs = zip(
                positions[node], np.add(positions[node], (np.asarray(headings[node]) * 50)))
            self.node_vector_collections.append(Arrow3D(
                xs, ys, zs,
                mutation_scale=2, lw=1,
                arrowstyle="-|>", color=self.plot_sphere_cm(colorval), alpha=self.vector_opacity
            ))
            # TODO Update to UPDATE DATA instead of re plotting
            self.plot_axes.add_artist(
                self.node_vector_collections[-1],
            )

    def redraw_fleet_heading_contribs(self):
        """


        """
        self._remove_vectors(self.node_contrib_collections)
        positions = self.ctl.get_fleet_positions(self.t)
        self._contrib_labels = []
        for node in range(self.ctl.get_n_vectors()):
            try:
                contributions = self.ctl.get_node_contribs(node, self.t)
            except Exception as e:
                self.log.error(
                    "something went bad, quitting Contrib display: {}".format(e))
                self.node_contrib_enabled = False
                break

            for contributor, contribution in contributions.iteritems():

                magnitude = mag(np.asarray(contribution))
                # Getting FPE's due to suspected zero vectors in mpl.draw
                if magnitude > 0.001:
                    xs, ys, zs = zip(
                        positions[node], np.add(positions[node], (np.asarray(contribution) * 50)))
                    vector = Arrow3D(
                        xs, ys, zs,
                        mutation_scale=2, lw=1,
                        arrowstyle="-|>", color=self.get_contrib_colour(contributor), alpha=self.contrib_opacity
                    )

                    self.node_contrib_collections.append(vector)
                    if contributor not in self._contrib_labels:
                        self._contrib_labels.append(contributor)
                    # TODO Update to UPDATE DATA instead of re plotting
                    self.plot_axes.add_artist(
                        self.node_contrib_collections[-1],
                    )
        self.plot_axes.legend(
            self.node_contrib_collections, self._contrib_labels, loc="lower left")
        self.plot_axes.autoscale()

    def get_contrib_colour(self, contrib_key):
        """

        :param contrib_key:
        :return: :raise ke:
        """
        try:
            return self._contrib_colour_dict[contrib_key]
        except (AttributeError, KeyError):
            self.contrib_colourmap = [
                cm.spectral(i) for i in np.linspace(0, 0.9, self.ctl.get_max_node_contribs())]
            self._contrib_colour_dict = {}
            for i, contrib_key in enumerate(self.ctl.get_contrib_keys()):
                self._contrib_colour_dict[
                    contrib_key] = self.contrib_colourmap[i]
            try:
                return self._contrib_colour_dict[contrib_key]
            except KeyError as ke:
                logging.error("CDict:%s" % self._contrib_colour_dict)
                raise ke("CDict:%s" % self._contrib_colour_dict)

    def _remove_sphere(self):
        if isinstance(self.sphere_line_collection, Line3DCollection) \
                and self.sphere_line_collection in self.plot_axes.collections:
            self.plot_axes.collections.remove(self.sphere_line_collection)

    @staticmethod
    def _remove_vectors(collection):
        try:
            length = len(collection)
        except TypeError:
            length = 1

        if collection is not None:
            for arrow in collection:
                if arrow is not None:
                    try:
                        arrow.remove()
                    except ValueError:
                        pass
        collection = [None for _ in range(length)]

    def update_metric_charts(self):
        # Cleanup Metrics
        """


        """
        while None in self.metric_views:
            ind = self.metric_views.index(None)
            del self.metric_axes[ind]
            del self.metric_views[ind]

        for i, plot in enumerate(self.metric_views):
            self.metric_axes[i] = plot.update(
                wanted=np.asarray(self.displayed_nodes))
            if self.metric_xlines[i] is not None:
                self.metric_xlines[i].set_xdata([self.t, self.t])
            else:
                self.metric_xlines[i] = self.metric_axes[
                    i].axvline(x=self.t, color='r', linestyle=':')
            if self.ctl.get_achievements() is None:
                pass
            elif self.achievement_xlines[i] is not None:
                self.achievement_xlines[i].set_xdata([self.t, self.t])
            else:
                for achievement in self.ctl.get_achievements():
                    self.achievement_xlines[i] = self.metric_axes[
                        i].axvline(x=achievement, color='b', alpha=0.1)
            if self._zoom_metrics:
                xlim = (max(0, self.t - 100), max(100, self.t + 100))
                self.metric_axes[i].set_xlim(xlim)
                self.metric_axes[i].set_ylim(*plot.ylim(xlim))
        self.metric_views[-1].ax.get_xaxis().set_visible(True)

    ####
    # Button Event Handlers
    ####
    def on_pause_btn(self, event):
        """

        :param event:
        """
        self.paused = not self.paused
        self.pause_btn.SetLabel("Resume" if self.paused else "Pause")

    def on_play_btn(self, event):
        """

        :param event:
        """
        self.paused = False
        wx.CallAfter(self.redraw_page, t=0)

    def on_faster_btn(self, event):
        """

        :param event:
        """
        self.d_t = int(min(max(2, self.d_t * 1.1), self.tmax / 20))
        self.log.debug("Setting time step to: %s" % self.d_t)

    def on_slower_btn(self, event):
        """

        :param event:
        """
        self.d_t = int(min(max(2, self.d_t * 0.9), self.tmax / 20))
        self.log.debug("Setting time step to: %s" % self.d_t)

    def on_time_slider(self, event):
        """

        :param event:
        """
        t = self.time_slider.GetValue()
        self.log.debug("Slider: Setting time to %d" % t)
        wx.CallAfter(self.redraw_page, t=t)

    ###
    # Display Selection Tools
    ###
    def on_sphere_chk(self, event):
        """

        :param event:
        """
        if not self.sphere_chk.IsChecked():
            self._remove_sphere()
            self.log.debug("Sphere Overlay Disabled")
            self.sphere_enabled = False
        else:
            self.log.debug("Sphere Overlay Enabled")
            self.sphere_enabled = True
        wx.CallAfter(self.redraw_page)

    def on_vector_chk(self, event):
        """

        :param event:
        """
        if not self.vector_chk.IsChecked():
            self._remove_vectors(self.node_vector_collections)
            self.log.info("Vector Overlay Disabled")
            self.node_vector_enabled = False
        else:
            self.log.info("Vector Overlay Enabled")
            self.node_vector_enabled = True
            if self.node_contrib_enabled:
                self.contrib_chk.SetValue(False)
                self.on_contrib_chk(event)
        wx.CallAfter(self.redraw_page)

    def on_contrib_chk(self, event):
        """

        :param event:
        """
        if not self.contrib_chk.IsChecked():
            self._remove_vectors(self.node_contrib_collections)
            self.log.info("Contrib Overlay Disabled")
            self.node_contrib_enabled = False
        else:
            self.log.info("Contrib Overlay Enabled")
            self.node_contrib_enabled = True
            if self.node_vector_enabled:
                self.vector_chk.SetValue(False)
                self.on_vector_chk(event)
        wx.CallAfter(self.redraw_page)

    def on_trail_slider(self, event):
        """

        :param event:
        """
        event.Skip()
        norm_trail = self.trail_slider.GetValue()
        self.trail_length = int(
            norm_trail * (self.ctl.get_final_tmax() / 100.0))
        self.log.debug("Slider: Setting trail to %d" % self.trail_length)
        wx.CallAfter(self.redraw_page)

    def on_metric_zoom_chk(self, event):
        """

        :param event:
        """
        if self.metric_zoom_chk.IsChecked():
            for axes in self.metric_axes:
                axes.autoscale_view(scalex=False, scaley=False, tight=True)
            self._zoom_metrics = True
        else:
            for plt, axes in zip(self.metric_views, self.metric_axes):
                axes.autoscale_view(scalex=True, scaley=True, tight=False)
                xlim = (0, self.tmax)
                axes.set_xlim(xlim)
                axes.set_ylim(*plt.ylim(xlim))
            self._zoom_metrics = False
        wx.CallAfter(self.redraw_page)

    def on_zoom_fleet_chk(self, event):
        # When unchecking, return to default viewpoint
        """

        :param event:
        """
        if not self.zoom_fleet_chk.IsChecked() and self._fleet_zoom:
            self.plot_axes.autoscale_view(
                scalex=True, scaley=True, scalez=True, tight=False)
            environment = self.ctl.get_extent()
            self.plot_axes.set_xlim3d(0, environment[0])
            self.plot_axes.set_ylim3d(0, environment[1])
            self.plot_axes.set_zlim3d(0, environment[2])
            self._fleet_zoom = False
        wx.CallAfter(self.redraw_page)

    ####
    # Menu Event Handlers
    ####
    def on_node_select(self, event):
        """
        Based on the wxPython demo - opens the MultiChoiceDialog
        and sets that selection as the node entries to be
        displayed
        :param event:
        """
        lst = list(self.ctl.get_vector_names())
        dlg = wx.MultiChoiceDialog(self,
                                   "Select nodes",
                                   "wx.MultiChoiceDialog", lst)

        selections = [
            i for i in range(self.ctl.get_n_vectors()) if self.displayed_nodes[i]]
        print selections
        dlg.SetSelections(selections)

        if dlg.ShowModal() == wx.ID_OK:
            selections = dlg.GetSelections()
            strings = [lst[x] for x in selections]
            print "You chose:" + str(strings)
            self.displayed_nodes.fill(False)
            for checked in selections:
                self.displayed_nodes[checked] = True

            self.update_metric_charts()

        wx.CallAfter(self.redraw_page)

    def on_idle(self, event):
        """

        :param event:
        """
        if not self.paused:
            if self._plot_initialised:
                self.move_T()
            elif self.ctl.model_is_ready():
                self.initialise_3d_plot()
            else:
                pass

    def on_resize(self, event):
        """
        Retrieve the Panel-size in pixels and update the plot panel, plot canvas, and figure.
        :param event:
        """
        plot_size = self.panel_sizer.GetChildren()[0].GetSize()
        self.log.info("plotsize:%s" % str(plot_size))

        self.plot_pnl.SetSize(plot_size)
        self.canvas.SetSize(plot_size)
        canvas_inches = (float(
            plot_size[0]) / self.fig.get_dpi(), float(plot_size[1]) / self.fig.get_dpi())
        self.log.debug("canvas_inch:%s" % str(canvas_inches))
        self.fig.set_size_inches(canvas_inches)
        wx.CallAfter(self.redraw_page)


class DriftingNavigator(VisualNavigator):
    def initialise_3d_plot(self):
        """


        """
        VisualNavigator.initialise_3d_plot(self)
        # Get Drift Plot info
        (xs, ys, zs) = self.ctl.get_3D_drift()
        self.drift_lines = [
            self.plot_axes.plot(x, y, z, linestyle=':', label=self.names[i], color=self.lines[i].get_color(), alpha=self.trail_opacity)[0] for
            i, (x, y, z) in enumerate(zip(xs, ys, zs))
        ]

    def redraw_page(self, t=None):
        """

        :param t:
        :raise:
        """
        VisualNavigator.redraw_page(self, t=t)
        for n, (line, label) in enumerate(zip(self.drift_lines, self.labels)):
            (xs, ys, zs) = self.ctl.get_3D_drift(
                node=n, time_start=self.t, length=self.trail_length)
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            line.set_label("({})".format(self.names[n]))
            try:
                label.set_position((xs[0], ys[0]))
                label.set_3d_properties(zs[0])
            except TypeError as err:
                self.log.error("x:%s,y:%s,z:%s" % (xs[0], ys[0], zs[0]))
                raise
            except IndexError:
                # In the case of the 'first time'. This should probably be removed in the case
                # of any architectural changes.
                pass


class ECEANavigator(DriftingNavigator):
    source = "intent"

    def initialise_3d_plot(self):
        """


        """
        DriftingNavigator.initialise_3d_plot(self)
        # Get Drift Plot info
        (xs, ys, zs) = self.ctl.get_3D_drift(source=self.source)
        self.intent_lines = [
            self.plot_axes.plot(x, y, z, linestyle=':', label=self.names[i], color=self.lines[i].get_color(), alpha=self.trail_opacity)[0] for
            i, (x, y, z) in enumerate(zip(xs, ys, zs))
        ]

    def redraw_page(self, t=None):
        """

        :param t:
        :raise:
        """
        DriftingNavigator.redraw_page(self, t=t)
        for n, (line, label) in enumerate(zip(self.intent_lines, self.labels)):
            (xs, ys, zs) = self.ctl.get_3D_drift(
                node=n, time_start=self.t, length=self.trail_length, source=self.source)
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            line.set_label("({})".format(self.names[n]))
            try:
                label.set_position((xs[0], ys[0]))
                label.set_3d_properties(zs[0])
            except TypeError as err:
                self.log.error("x:%s,y:%s,z:%s" % (xs[0], ys[0], zs[0]))
                raise
            except IndexError:
                # In the case of the 'first time'. This should probably be removed in the case
                # of any architectural changes.
                pass
