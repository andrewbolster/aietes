import SimPy.Simulation as Sim
import logging

import Layercake

module_logger = logging.getLogger('AIETES')

class Simulation():
    """
    Defines a single simulation
    """
    nodes=[]
    fleets=[]
    def __init__(self,config_file):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))o
        self.logger.info('creating instance')

        Sim.initialize()

        if not hasattr(config_file,'channel_event_name'):
            config_file['channel_event_name']='AcousticEvent'

        self.channel_event = Sim.SimEvent(config_file['channel_event_name'])

        try:
            self.config = self.validateConfig(config_file) #TODO JSON Lint?
            self.environment = configureEnvironment()
            self.nodes = configureNodes()
        except ConfigError as err: #TODO

    def configureEnvironment(self, host, config=self.config):
        """ Should be called from each node
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        pass

    def configureLayercake(self,config=self.config):
        """
        Configure the 5-layer comms model based on the given config file.
        Assume defaults of stupid ALOHA
        """
        pass

    def configureNodes(self,config=self.config):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        pass

class Six_vector():
    """
    Class that implements 6-degrees-of-freedom position/orientation tuples and their basic operations
    """
    def __init__(self,seq)
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))o
        self.logger.info('creating instance')
        (x,y,z,a,b,g)=seq

class Node(Sim.process):
    """
    Generic Representation of a network node
    """
    def __init__(self,name,simulation):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))o
        self.logger.info('creating instance')
        Sim.Process.__init__(self,name=name)
        self.simulation=simulation
        # Comms Stack
        self.layers = self.simulation.getLayerCake(self)

    def lifecycle(self):
        """
        Called to update internal awareness and motion
        """
        while True:

            yield








