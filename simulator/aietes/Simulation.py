import SimPy.Simulation as Sim
import logging
import numpy
import scipy
from Layercake import Layercake
import MAC, Net, Applications
from Environment import Environment
from Node import Node

from configobj import ConfigObj
import validate

import random

module_logger = logging.getLogger('AIETES')

naming_convention=['Zeus','Neptune','Jupiter','Hera','Mercury','Faunus','Hestia','Demeter','Poseidon','Diana','Apollo','Pluto','Juno','Vesta','Diana','Hermes','Persephone','Cupid','Ares','Hephaestus']

class dotdictify(dict):
    marker = object()
    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError, 'expected dict'

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, dotdictify):
            value = dotdictify(value)
        dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        found = self.get(key, dotdictify.marker)
        if found is dotdictify.marker:
            found = dotdictify()
            dict.__setitem__(self, key, found)
        return found

    __setattr__ = __setitem__
    __getattr__ = __getitem__

class ConfigError(Exception):
    """
    Raised when a configuration cannot be validated through ConfigObj/Validator
    Contains a 'status' with the boolean dict representation of the error
    """
    def __init__(self,value):
        module_logger.critical("Invalid Config; Dying")
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
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')

        #Attempt Validation and construct the simulation from that config.
        try:
            self.validateConfig(config_file)
        except ConfigError as err:
            self.logger.error("Error in configuration, cannot continue: %s"%err)

        #Initialise simulation environment and configure a global channel event
        Sim.initialize()
        self.channel_event = Sim.SimEvent(self.config.sim['channel_event_name'])

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
        Sim.simulate(until=self.config.sim['sim_duration'])

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

        self.config= dotdictify(self.config)
        # NODE CONFIGURATION
        if self.config.Nodes.count > len(naming_convention):
            # If the naming convention can't provide unique names, bail
            raise ConfigError("Not Enough Names in dictionary for number of nodes requested!")

        if bool(self.config.Nodes.node_names):
            # If given the correct number of names in the config file, do nothing
            if self.config.Nodes.count > len(self.config.Nodes.node_names):
                raise ConfigError("Not Enough Names in configfile for number of nodes requested!")
        else:
            # Otherwise make up names from the naming_convention
            for n in range(self.config.Nodes.count):
                candidate_name= random.choice(naming_convention)
                while candidate_name in [ x.name for x in self.nodes ]:
                    candidate_name= random.choice(naming_convention)
                self.config.Nodes.node_names+=candidate_name

        # LAYERCAKE CONFIG CHECK
        try:
            self.config.mac_mod=getattr(MAC,str(self.config.MAC.type))
        except AttributeError:
            raise ConfigError("Can't find MAC: %s"%self.config.MAC.type)

        try:
            self.config.net_mod=getattr(Net,str(self.config.Network.type))
        except AttributeError:
            raise ConfigError("Can't find Network: %s"%self.config.Network.type)

        try:
            self.config.app_mod=getattr(Applications,str(self.config.Application.type))
        except AttributeError:
            raise ConfigError("Can't find Application: %s"%self.config.Application.type)

    def configureEnvironment(self, config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        self.environment=Environment(shape=config.shape,
                                     resolution=config.resolution,
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
