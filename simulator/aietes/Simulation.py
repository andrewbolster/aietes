import SimPy.Simulation as Sim
import logging
import numpy
import scipy
import Layercake
import environment
import defaults

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

        #Attempt Validation and construct the simulation from that config.
        try:
            self.config = self.validateConfig(config_file)
        except ConfigError as err:
            self.logger.err("Error in configuration, cannot continue: %s"%err)

        #Initialise simulation environment and configure a global channel event
        Sim.initialize()
        self.channel_event = Sim.SimEvent(config_file.sim['channel_event_name'])

        self.environment = configureEnvironment()
        self.nodes = configureNodes()

    def validateConfig(config_file, defaults=defaults):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults
        """
        

    def configureEnvironment(self, host, config=self.config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        self.environment=Environment(shape=config.shape,
                                     scale=config.scale,
                                     base_depth=config.base_depth
                                    )

    def configureLayercake(self,config=self.config):
        """
        Configure the 5-layer comms model based on the given config file.
        Assume defaults of stupid ALOHA
        """
        pass

    def configureNodes(self,config=self.config,fleet=None):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        #Configure specified nodes
        #TODO config checker should generate node_names set
        for nodeName in config.node_names:
            new_node=Node(
                nodeName,
                self,
                vector=self.vectorGen(
                    nodeName
                )
            )
            self.nodes.append(new_node)

        #TODO Node Field Configurations based on generative behaviours
        #TODO Fleet implementation

    def vectorGen(self,nodeName):
        """
        If a node is named in the configuration file, use the defined initial vector
        otherwise, use configured behaviour to assign an initial vector
        """
        if hasattr(self.config.init_node_vectors,nodeName):
            vector = self.config.init_node_vectors[nodeName]
            self.logger.info("Gave node %s a configured initial vector: %s"%(nodeName,str(vector)))
            return
        else:
            #TODO add additional behaviours, eg random within enviroment; distribution around a point, etc
            self.logger.debug("Gave node %s a zero vector"%nodeName)
            return [0,0,0,0,0,0]







