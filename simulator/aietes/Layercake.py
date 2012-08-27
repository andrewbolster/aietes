import PHY, MAC, Net, Applications
from Tools import Sim
import logging
import pydot

class Layercake():
    """
    Defines the Four relevant network layers for a given node
    PHY,MAC,Network,Application
    """
    def __init__(self, host, simulation,fake=False,):
        self.host=host
        self.channel_event=simulation.channel_event
        self.config= simulation.config
        self.logger = host.logger

        #PHY
        self.phy = PHY.PHY(self,self.channel_event,self.config.PHY)
        #MAC
        self.mac = self.config.mac_mod(self,self.config.MAC)
        #Routing
        self.net = self.config.net_mod(self,self.config.Network)
        #Application
        self.app = self.host.config.app_mod(self,self.config.Nodes.Application)

    def activate(self):
        """
        Fired on Sim Start
        """
        self.mac.activate()
        self.app.activate()

