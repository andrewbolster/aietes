__author__ = 'andrewbolster'
import wx


class Simulator(wx.Panel):
    def __init__(self, parent, frame, *args, **kw):
        wx.Panel.__init__(self, parent, *args, **kw)
        wx.StaticText(self, wx.ID_ANY, "This is the Simulator", (20, 20))


    def on_idle(self, event):
        pass

