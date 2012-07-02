import SimPy.Simulation as Sim
import logging
import numpy
import scipy
import Layercake
import Environment

from configobj import ConfigObj

module_logger = logging.getLogger('AIETES')

naming_convention=['Zeus','Neptune','Jupiter','Hera','Mercury','Faunus','Hestia','Demeter','Poseidon','Diana','Apollo','Pluto','Juno','Vesta','Diana','Hermes','Persephone','Cupid','Ares','Hephaestus']

class SimConfig(dict):
    def __getattr__(self, attr):
        return self.get(attr, None)
    __setattr__= ConfigObj.__setitem__
    __delattr__= ConfigObj.__delitem__

class Simulation():
    """
    Defines a single simulation
    """
    nodes=[]
    fleets=[]
    def __init__(self,config_file):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')

        #Attempt Validation and construct the simulation from that config.
        try:
            self.config = self.validateConfig(config_file)
        except ConfigError as err:
            self.logger.err("Error in configuration, cannot continue: %s"%err)

        #Initialise simulation environment and configure a global channel event
        Sim.initialize()
        self.channel_event = Sim.SimEvent(config.sim['channel_event_name'])

        self.environment = configureEnvironment()
        self.nodes = configureNodes()

        #Configure Resources
        self.update_flag= Sim.Resource(
            capacity= len(self.nodes),
            name= 'Update Flag')
        self.process_flag= Sim.Resource(
            capacity= len(self.nodes),
            name= 'Process Flag')
        self.move_flag= Sim.Resource(
            capacity= len(self.nodes),
            name= 'Move Flag')

    def go(self):
        Sim.simulate(until=self.config.sim['sim_duration'])

    def clearToStep(self):
        n_nodes=len(self.nodes)
        return self.update_flag.n == n_nodes \
                and self.process_flag.n == n_nodes \
                and self.move_flag.n == n_nodes

    def validateConfig(self,config_file, configspec):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults
        """
        self.config= SimConfig(config_file)
        assert config.Nodes.count < len(naming_convention), "Not Enough Names!"
        for n in range(config.Nodes.count):
            candidate_name= random.choice(naming_convention)
            while candidate_name in [ x.name for x in self.nodes ]:
                candidate_name= random.choice(naming_convention)



    def configureEnvironment(self, host, config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        self.environment=Environment(shape=config.shape,
                                     scale=config.scale,
                                     base_depth=config.base_depth
                                    )

    def configureNodes(self,config,fleet=None):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        #Configure specified nodes
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
