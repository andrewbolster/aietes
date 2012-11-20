#!/usr/bin/env python

import wx
import os
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)

import matplotlib
matplotlib.use('WXAgg')

from bounos import DataPackage

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

_ROOT=os.path.abspath(os.path.dirname(__file__))


class EphyraFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(EphyraFrame, self).__init__(*args, **kw)
        self.log = logging.getLogger(self.__module__)
        parser = argparse.ArgumentParser(description="GUI simulation and Analysis Suite for the Aietes framework")

        parser.add_argument('-o','--open',
                            dest='data_file', action='store', default=None,
                            metavar='XXX.npz',
                            help='Aietes DataPackage to be analysed'
                           )

        self.args = parser.parse_args()

        self.paused = True
        self.t = 0

        self.CreateMenuBar()
        panel = wx.Panel(self)

        self.plot_pnl = wx.Panel(self)
        self.plot_pnl.SetBackgroundColour(wx.BLACK)
        self.control_pnl = wx.Panel(self)

        self.time_slider = wx.Slider(self.control_pnl, value=1, minValue=0, maxValue=1)
        self.pause_btn = wx.Button(self.control_pnl, label="Pause")
        self.play_btn  = wx.Button(self.control_pnl, label="Play")
        self.forw_btn  = wx.Button(self.control_pnl, label=">>")
        self.back_btn  = wx.Button(self.control_pnl, label="<<")
        self.slider2 = wx.Slider(self.control_pnl, value=1, minValue=0, maxValue=100, size=(120, -1))

        self.Bind(wx.EVT_SCROLL, self.on_time_slider, self.time_slider)
        self.Bind(wx.EVT_BUTTON, self.on_pause_btn, self.pause_btn)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_pause_btn, self.pause_btn)
        self.Bind(wx.EVT_BUTTON, self.on_play_btn, self.play_btn)
        self.Bind(wx.EVT_BUTTON, self.on_forw_btn, self.forw_btn)
        self.Bind(wx.EVT_BUTTON, self.on_back_btn, self.back_btn)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        hbox1.Add(self.time_slider, proportion=1)
        hbox2.Add(self.pause_btn)
        hbox2.Add(self.play_btn, flag=wx.RIGHT, border=5)
        hbox2.Add(self.forw_btn, flag=wx.LEFT, border=5)
        hbox2.Add(self.back_btn)
        hbox2.Add(self.slider2, flag=wx.TOP|wx.LEFT, border=5)

        vbox.Add(hbox1, flag=wx.EXPAND|wx.BOTTOM, border=10)
        vbox.Add(hbox2, proportion=1, flag=wx.EXPAND)
        self.control_pnl.SetSizer(vbox)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.plot_pnl, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(self.control_pnl, flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=10)

        self.Bind(wx.EVT_SIZE, self.on_resize)

        self.SetMinSize((800, 600))
        self.CreateStatusBar()
        self.SetSizer(self.sizer)

        self.SetSize((350, 200))
        self.SetTitle('Player')
        self.Centre()
        self.Show(True)

        if self.args.data_file is not None:
            if os.path.exists(self.args.data_file):
                self.log.info("Loading data from %s"%self.args.data_file)
                self.load_data(self.args.data_file)
            else :
                self.log.error("File Not Found: %s"%self.args.data_file)

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
        self.fig = Figure()
        self.canvas = FigureCanvas(self.plot_pnl, -1, self.fig)
        self.axes = self.fig.add_axes([0,0,1,1], projection='3d')

        # Find initial display state for viewport
        self.lines = [ self.axes.plot( xs, ys, zs)[0] for xs,ys,zs in self.data.p ]
        for n,line in enumerate(self.lines):
            line.set_label(self.data.names[n])

        shape = self.data.environment
        self.axes.legend()
        self.axes.set_title("Tracking overview of %s"%self.data.title)
        self.axes.set_xlim3d((0,shape[0]))
        self.axes.set_ylim3d((0,shape[1]))
        self.axes.set_zlim3d((0,shape[2]))
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.set_zlabel('Z')

    def redraw_plot(self):
        for n,line in enumerate(self.lines):
            (xs,ys,zs)=self.data.trail_of(n, self.t)
            line.set_data(xs,ys)
            line.set_3d_properties(zs)
            line.set_label(self.data.names[n])

        self.canvas.draw()

    def resize_panel(self):
        plot_size = self.sizer.GetChildren()[0].GetSize()
        self.plot_pnl.SetSize(plot_size)
        self.canvas.SetSize(plot_size)
        self.fig.set_size_inches( float( plot_size[0])/self.fig.get_dpi(),
                                  float( plot_size[0])/self.fig.get_dpi())



    def load_data(self, data_file):
        """ Load an external data file for plotting and navigation

        :param data_file: Aietes generated data for processing
        :type data_file: str

        Configures Plot area and adjusts control area
        """

        self.data = DataPackage(data_file)
        self.init_plot()
        self.log.debug("Successfully loaded data from %s, containg %d nodes over %d seconds"%(
            self.data.title,
            self.data.n,
            self.data.tmax
            )
        )
        self.time_slider.SetRange(0,self.data.tmax)
        self.time_slider.SetValue(self.data.tmax)

    def on_close(self, event):
        dlg = wx.MessageDialog(self,
            "Do you really want to close this application?",
            "Confirm Exit", wx.OK|wx.CANCEL|wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.Destroy()

    def on_open(self,event):
        dlg = wx.FileDialog(
                        self, message="Select a DataPackage",
                        defaultDir=os.getcwd(),
                        wildcard="*.npz",
                        style=wx.OPEN | wx.CHANGE_DIR
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
        label = "Resume" if self.paused else "Pause"
        self.pause_btn.SetLabel(label)

    def on_play_btn(self,event):
        pass

    def on_forw_btn(self,event):
        pass

    def on_back_btn(self,event):
        pass

    def on_time_slider(self,event):
        event.Skip()
        self.t = self.time_slider.GetValue()
        self.log.debug("Slider: Setting time to %d"%self.t)
        wx.CallAfter(self.redraw_plot)
        pass

    def on_resize(self,event):
        event.Skip()
        wx.CallAfter(self.resize_panel)

def main():
    app = wx.PySimpleApp()
    app.frame = EphyraFrame(None)
    app.frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

