from SimPy import Simulation as Sim
import PHY, MAC, Network, Application
import logging
import pydot
module_logger=logging.getLogger('AIETES.Layercake')

class Layercake():
    """
    Defines the Four relevant network layers for a given node
    PHY,MAC,Network,Application
    """
    def __init__(self,config):
    
        #PHY
        self.phy = simulation.getPHY(self)
        #MAC
        self.mac = simulation.getMAC(self)
        #Routing
        self.net = simluation.getNet(self)
        #Application
        self.app = simulation.getApp(self)



