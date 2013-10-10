#!/usr/bin/env python
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
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

__author__ = 'andrewbolster'

import os
import logging
import traceback
import sys

import wx

import matplotlib
import matplotlib.backends.backend_wxagg

from mpl_toolkits.mplot3d import proj3d

from matplotlib.patches import FancyArrowPatch


matplotlib.rcParams.update({'font.size': 8})

import numpy as np


class MetricView():
    """
    This Class is a plotable view of the Metric class availiable from Bounos.
    It is instantiated with the representative Bounos.Metric base metric
    """

    def __init__(self, axes, base_metric, *args, **kw):
        self.ax = axes
        self.data = base_metric.data.view()
        self.label = base_metric.label
        self.highlight_data = base_metric.highlight_data
        self.ndim = 0
        self.last_wanted = np.asarray([])
        if __debug__: logging.debug("%s" % self)

    def plot(self, wanted=None, time=None):
        """
        Update the Plot based on 'new' wanted data
        """
        self.ax.clear()
        self.ax.set_ylabel(self.label)
        self.ax.get_xaxis().set_visible(True)

        if all(wanted == True) or self.ndim == 1:
            self.ax.plot(self.data, alpha=0.3)
        else:
            logging.info("Printing %s with Wanted:%s" % (self, wanted))
            self.ax.plot(np.ndarray(buffer=self.data, shape=self.data.shape)[:, wanted], alpha=0.3)

        if self.highlight_data is not None:
            self.ax.plot(self.highlight_data, color='k', linestyle='--')

        self.last_wanted = wanted
        self.ax.relim()
        return self.ax

    def update(self, wanted=None, time=None):
        if wanted is None: wanted = []
        if np.array_equiv(wanted, self.last_wanted):
            #Assume that the ylim will sort scoping out later...
            return self.ax
        else:
            #Something has changed so recalculate
            self.last_wanted = wanted
            return self.plot(wanted, time)

    def ylim(self, xlim, margin=None):
        (xmin, xmax) = xlim
        if self.highlight_data is not None:
            data = np.append(self.data, self.highlight_data).reshape((self.data.shape[0], -1))
        else:
            data = self.data

        try:
            if self.ndim > 1:
                slice = data[xmin:xmax][:]
            else:
                slice = data[xmin:xmax]
            ymin = slice.min()
            ymax = slice.max()
            range = ymax - ymin
            if margin is None:
                margin = range * 0.3333

            ymin -= margin
            ymax += margin
        except ValueError as e:
            logging.critical("Data:%s, XLim:%s" % (data, xlim))
            raise e
        return ymin, ymax


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
        wx.Frame.__init__(self, None, title=EphyraNotebook.description, *args, **kw)
        self.log = logging.getLogger(self.__module__)
        self.ctl = controller
        p = VisualNavigator(self, self, wx.ID_ANY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(p, proportion=1, flag=wx.GROW)
        self.SetMinSize((800, 600))
        self.SetSizer(sizer)
        sizer.Fit(p)
        self.Layout()


from ephyra.Views.VisualNavigator import VisualNavigator
from ephyra.Views.Configurator import Configurator
from ephyra.Views.Simulator import Simulator


class EphyraNotebook(wx.Frame):
    def __init__(self, controller, *args, **kw):
        wx.Frame.__init__(self, None, title="Ephyra")
        self.log = logging.getLogger(self.__module__)
        self.ctl = controller
        self.args = kw.get("exec_args", None)

        # Create a panel and a Notebook on the Panel
        self.p = wx.Panel(self)
        self.nb = wx.Notebook(self.p)
        self.CreateMenuBar()

        self.Bind(wx.EVT_MAXIMIZE, self.on_resize)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_IDLE, self.on_idle)

        pages = [VisualNavigator, Configurator, Simulator]
        pages = [VisualNavigator]

        for page in pages:
            self.log.debug("Adding Page: %s" % page.__name__)
            self.nb.AddPage(page(self.nb, self), page.__name__)

        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetFieldsCount(3)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.nb, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.SetAutoLayout(1)
        self.p.SetSizer(self.sizer)
        self.SetMinSize((800, 600))
        self.Layout()
        self.Show()
        self.nb.SetSelection(0)


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
        self.p.Layout()
        current_page = self.nb.GetCurrentPage()
        try:
            current_page.on_resize(event)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            wx.CallAfter(self.exit)
            raise e

    ####
    # File Events
    ####
    def on_new(self, event):
        dlg = wx.MessageDialog(self,
                               message="This will start a new simulation using the SimulationStep system to generate results in 'real' time and will be fucking slow",
                               style=wx.OK | wx.CANCEL | wx.ICON_EXCLAMATION
        )
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.ctl.new_simulation()

    def on_open(self, event):
        dlg = wx.FileDialog(
            self, message="Select a DataPackage",
            defaultDir=os.getcwd(),
            wildcard="*.npz",
            style=wx.OPEN | wx.CHANGE_DIR
        )

        if dlg.ShowModal() == wx.ID_OK:
            data_path = dlg.GetPaths()
            if len(data_path) > 1:
                self.log.warn("Too many paths given, only taking the first anyway")
            data_path = data_path[0]
            self.ctl.load_data_file(data_path)

    ###
    # Status Management
    ###
    def update_status(self, msg):
        self.status_bar.SetStatusText(msg, number=0)
        if msg is not "Idle":
            self.log.info(msg)

    def update_timing(self, t, tmax):
        self.status_bar.SetStatusText("%s:%d/%d" % ("SIM" if self.ctl.is_simulation() else "REC", t, tmax), number=2)
