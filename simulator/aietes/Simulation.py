import SimPy.Simulation as Sim

import logging

class Simulation():
    """
    Defines a single simulation
    """
    
    def __init__(self,config_file):

class Layercake():
    """
    Defines the Four relevant network layers for a given node
    PHY,MAC,Network,Application
    """
    def __init__(self,simulation)
        #PHY
        self.phy = simulation.getPHY(self)
        #MAC
        self.mac = simulation.getMAC(self)
        #Routing
        self.net = simluation.getNet(self)
        #Application
        self.app = simulation.getApp(self)

class Six_vector():
    """
    Class that implements 6-degrees-of-freedom position/orientation tuples and their basic operations
    """

class Node(Sim.process):
    """
    Generic Representation of a network node
    """
    def __init__(self,name,simulation):
        Sim.Process.__init__(self,name=name)
        self.simulation=simulation
        # Comms Stack
        self.layers = self.simulation.getLayerCake(self,self.simulation)

    def lifecycle(self):
        """
        Called to update internal awareness
        """
        while True:
            yield #TODO








