#!/usr/bin/env python

import wx
import os

from bounos import DataPackage

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure

_ROOT=os.path.abspath(os.path.dirname(__file__))


class EphyraFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(EphyraFrame, self).__init__(*args, **kw)

        self.InitUI()

        self.paused = True

    def InitUI(self):
        self.CreateMenuBar()
        panel = wx.Panel(self)

        pnl1 = wx.Panel(self)
        pnl1.SetBackgroundColour(wx.BLACK)
        pnl2 = wx.Panel(self)

        self.time_slider = wx.Slider(pnl2, value=0, minValue=0, maxValue=1)
        self.pause_btn = wx.Button(pnl2, label="Pause")
        self.play_btn  = wx.Button(pnl2, label="Play")
        self.forw_btn  = wx.Button(pnl2, label=">>")
        self.back_btn  = wx.Button(pnl2, label="<<")
        self.slider2 = wx.Slider(pnl2, value=1, minValue=0, maxValue=100, 
            size=(120, -1))

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
        pnl2.SetSizer(vbox)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(pnl1, proportion=1, flag=wx.EXPAND)
        sizer.Add(pnl2, flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=10)

        self.SetMinSize((800, 600))
        self.CreateStatusBar()
        self.SetSizer(sizer)

        self.SetSize((350, 200))
        self.SetTitle('Player')
        self.Centre()
        self.Show(True)

    def CreateMenuBar(self):

        menubar = wx.MenuBar()
        filem = wx.Menu()
        play = wx.Menu()
        view = wx.Menu()
        tools = wx.Menu()
        favorites = wx.Menu()
        help = wx.Menu()

        openm = filem.Append(wx.ID_ANY, '&open', 'Open a datafile')
        self.Bind(wx.EVT_MENU, self.on_open, openm)
        exitm = filem.Append(wx.ID_ANY, '&quit', 'Quit application')
        self.Bind(wx.EVT_MENU, self.on_close, exitm)

        menubar.Append(filem, '&File')
        menubar.Append(play, '&Play')
        menubar.Append(view, '&View')
        menubar.Append(tools, '&Tools')
        menubar.Append(favorites, 'F&avorites')
        menubar.Append(help, '&Help')

        self.SetMenuBar(menubar)

    def init_plot(self):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111, projection='3d')

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
        for n,line in enumerate(lines):
            (xs,ys,zs)=data.trail_of(n, self.t)
            line.set_data(xs,ys)
            line.set_3d_properties(zs)
            line.set_label(data.names[n])

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
            self.data = DataPackage(dlg.GetPaths()[0])
            self.init_plot()
        #TODO time_slider.setrange(min,max)

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
        pass


def main():
    app = wx.PySimpleApp()
    app.frame = EphyraFrame(None)
    app.frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

