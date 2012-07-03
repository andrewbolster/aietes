import SimPy.Simulation as Sim
import logging
import numpy
import scipy

from configobj import ConfigObj
import validate
import random

from Layercake import Layercake
import MAC, Net, Applications
from Environment import Environment
from Node import Node
import Behaviour

from Tools import dotdict, baselogger

naming_convention=['Zeus','Neptune','Jupiter','Hera','Mercury','Faunus','Hestia','Demeter','Poseidon','Diana','Apollo','Pluto','Juno','Vesta','Diana','Hermes','Persephone','Cupid','Ares','Hephaestus']


class ConfigError(Exception):
    """
    Raised when a configuration cannot be validated through ConfigObj/Validator
    Contains a 'status' with the boolean dict representation of the error
    """
    def __init__(self,value):
        baselogger.critical("Invalid Config; Dying")
        self.status=value
    def __str__(self):
        return repr(self.status)

class Simulation():
    """
    Defines a single simulation
    """
    nodes=[]
    fleets=[]
    def __init__(self,config_file=''):
        self.logger = logging.getLogger("%s.%s"%(baselogger.name,self.__class__.__name__))
        self.logger.info('creating instance')

        #Attempt Validation and construct the simulation from that config.
        try:
            self.validateConfig(config_file)
        except ConfigError as err:
            self.logger.error("Error in configuration, cannot continue: %s"%err)
            raise SystemExit(1)

        #Initialise simulation environment and configure a global channel event
        Sim.initialize()
        self.channel_event = Sim.SimEvent(self.config.Simulation.channel_event_name)

        self.environment = self.configureEnvironment(self.config.Environment)
        self.nodes = self.configureNodes(self.config.Nodes)

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
        Sim.simulate(until=self.config.Simulation.sim_duration)

    def clearToStep(self):
        n_nodes=len(self.nodes)
        return self.update_flag.n == n_nodes \
                and self.process_flag.n == n_nodes \
                and self.move_flag.n == n_nodes

    def validateConfig(self,config_file, configspec='defaults.conf'):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults
        """
        # GENERIC CONFIG CHECK
        self.config= ConfigObj(config_file,configspec=configspec)
        config_status = self.config.validate(validate.Validator(),copy=True)
        if not config_status:
            # If configspec doesn't match the input, bail
            raise ConfigError(config_status)

        self.config= dotdict(self.config.dict())
        # NODE CONFIGURATION
        if self.config.Nodes.count > len(naming_convention):
            # If the naming convention can't provide unique names, bail
            raise ConfigError("Not Enough Names in dictionary for number of nodes requested:%s/%s!"%(self.config.Nodes.count,len(naming_convention)))

        if bool(self.config.Nodes.node_names):
            # If given the correct number of names in the config file, do nothing
            if self.config.Nodes.count > len(self.config.Nodes.node_names):
                raise ConfigError("Not Enough Names in configfile for number of nodes requested!")
            assert int(self.config.Nodes.count) == len(self.config.Nodes.node_names)
        else:
            # Otherwise make up names from the naming_convention
            assert self.config.Nodes.count>=1, "No nodes configured"
            for n in range(self.config.Nodes.count):
                candidate_name= naming_convention[numpy.random.randint(0,len(naming_convention))]
                while candidate_name in [ x.name for x in self.nodes ]:
                    candidate_name= naming_convention[numpy.random.randint(0,len(naming_convention))]
                self.config.Nodes.node_names.append(candidate_name)
                self.logger.info("Gave node %d name %s"%(n,candidate_name))

        # LAYERCAKE CONFIG CHECK
        try:
            self.config.mac_mod=getattr(MAC,str(self.config.MAC.protocol))
        except AttributeError:
            raise ConfigError("Can't find MAC: %s"%self.config.MAC.protocol)

        try:
            self.config.net_mod=getattr(Net,str(self.config.Network.protocol))
        except AttributeError:
            raise ConfigError("Can't find Network: %s"%self.config.Network.protocol)

        try:
            self.config.app_mod=getattr(Applications,str(self.config.Application.protocol))
        except AttributeError:
            raise ConfigError("Can't find Application: %s"%self.config.Application.protocol)

        

        #Confirm
        self.logger.info("Built Config: %s"%str(self.config))

    def configureEnvironment(self, config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        return Environment(shape=config.shape,
                           resolution=config.resolution,
                           base_depth=config.base_depth
                          )

    def configureNodes(self,config,fleet=None):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        nodelist = []
        #Configure specified nodes
        for nodeName in config.node_names:
            new_node=Node(
                nodeName,
                self,
                config=self.vectorGen(
                    nodeName
                )
            )
            nodelist.append(new_node)

        #TODO Fleet implementation

        return nodelist

    def vectorGen(self,nodeName):
        """
        If a node is named in the configuration file, use the defined initial vector
        otherwise, use configured behaviour to assign an initial vector
        """
        node_config = self.config.Nodes
        try:# If there is an entry, use it
            vector = node_config.initial_node_vectors[nodeName]
            self.logger.info("Gave node %s a configured initial vector: %s"%(nodeName,str(vector)))
        except KeyError:
            #TODO add additional behaviours, eg random within enviroment; distribution around a point, etc
            gen_style = node_config.vector_generation
            if gen_style == "random":
                vector = self.environment.random_position()
                self.logger.debug("Gave node %s a random vector: %s"%(nodeName,vector))
            else:
                vector = [0,0,0]
                self.logger.debug("Gave node %s a zero vector from %s"%(nodeName,gen_style))
        assert len(vector)==3, "Incorrectly sized vector"
        # BEHAVIOUR CONFIG CHECK
        try:
            behave_mod=getattr(Behaviour,str(node_config.protocol))
        except AttributeError:
            raise ConfigError("Can't find Behaviour: %s"%node_config.protocol)

        config = dotdict({  'vector':vector,
                            'max_speed':node_config.max_speed,
                            'max_turn':node_config.max_turn,
                            'behaviour':behave_mod
                        })
        return config
