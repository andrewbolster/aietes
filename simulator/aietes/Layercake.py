from SimPy import Simulation as Sim
import PHY, MAC, Net, Application
import logging
import pydot
module_logger=logging.getLogger('AIETES.Layercake')

class Layercake():
    """
    Defines the Four relevant network layers for a given node
    PHY,MAC,Network,Application
    """
    def __init__(self,host,simulation):
        self.host=host
        self.channel_event=simulation.channel_event

        #PHY
        self.phy = PHY.PHY(self,self.channel_event)
        #MAC
        self.mac = MAC.ALOHA(self)
        #Routing
        self.net = Net.RoutingTable(self)
        #Application
        self.app = Application(self)



