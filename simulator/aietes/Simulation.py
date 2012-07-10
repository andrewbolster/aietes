import SimPy.Simulation as Sim
import logging
import numpy as np
import scipy

from configobj import ConfigObj
import validate
import random

from Layercake import Layercake
import MAC, Net, Applications
from Environment import Environment
from Node import Node
import Behaviour

from Tools import *

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3
import matplotlib.animation as ani
from operator import itemgetter


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
class Fleet(Sim.Process):
    """
    Fleets act initially as traffic managers for Nodes
    """
    def __init__(self,nodes):

        self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
        self.logger.info("creating instance")
        Sim.Process.__init__(self,name="Fleet")
        self.nodes = nodes

    def activate(self):
        Sim.activate(self,self.lifecycle())
        for node in self.nodes:
            node.activate()


    def lifecycle(self):
        def allPassive():
            return all([n.passive() for n in self.nodes])
        self.logger.info("Initialised Node Lifecycle")
        while(True):
            yield Sim.waituntil, self, allPassive
            self.logger.info("Fleet Step")
            for node in self.nodes:
                Sim.reactivate(node)


class Simulation():
    """
    Defines a single simulation
    """
    def __init__(self,config_file=''):
        self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
        self.logger.info("creating instance from %s"%config_file)
        self.config_file=config_file
        self.nodes=[]
        self.fleets=[]

    def prepare(self):
        #Attempt Validation and construct the simulation from that config.
        try:
            self.config=self.validateConfig(self.config_file)
            baselogger.setLevel(LOGLEVELS.get(self.config.log_level,logging.NOTSET))
        except ConfigError as err:
            self.logger.error("Error in configuration, cannot continue: %s"%err)
            raise SystemExit(1)

        #Initialise simulation environment and configure a global channel event
        Sim.initialize()
        self.channel_event = Sim.SimEvent(self.config.Simulation.channel_event_name)

        self.environment = self.configureEnvironment(self.config.Environment)
        self.nodes = self.configureNodes(self.config.Nodes)

        #Single Fleet to control all
        self.fleets.append(Fleet(self.nodes))

        # Set up 'join-like' operation for nodes
        self.move_flag = Sim.Resource(capacity=len(self.nodes))
        self.process_flag = Sim.Resource(capacity=len(self.nodes))

    def inner_join(self):
        """
        When all nodes have a move flag and none are processing
        """
        joined=self.move_flag.n == 0 and self.process_flag.n == len(self.nodes)
        if joined:
            self.logger.debug("Joined: %s,%s"%(self.move_flag.n,self.process_flag.n))
        return joined

    def outer_join(self):
        """
        When all nodes have a processing flag and none are moving
        """
        joined=self.move_flag.n == len(self.nodes) and self.process_flag.n == 0
        if joined:
            self.logger.debug("Joined: %s,%s"%(self.move_flag.n,self.process_flag.n))
        return joined

    def simulate(self):
        """
        Initiate the processed Simulation
        """
        duration_intervals = self.config.Simulation.sim_duration/self.config.Simulation.sim_interval
        self.logger.info("Initialising Simulation, to run for %s steps"%duration_intervals)
        for fleet in self.fleets:
            fleet.activate()

        Sim.simulate(until=duration_intervals)

    def reverse_node_lookup(self, uuid):
        """Return Node Given UUID
        """
        for n in self.nodes:
            if n.id == uuid:
                return n
        raise KeyError("Given UUID does not exist in Nodes list")


    def validateConfig(self,config_file='', configspec='defaults.conf'):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults
        """
        # GENERIC CONFIG CHECK
        config= ConfigObj(config_file,configspec=configspec)
        config_status = config.validate(validate.Validator(),copy=True)
        if not config_status:
            # If configspec doesn't match the input, bail
            raise ConfigError(config_status)

        config= dotdict(config.dict())
        naming_convention= config.Nodes.naming_convention
        # NODE CONFIGURATION
        if config.Nodes.count > len(naming_convention):
            # If the naming convention can't provide unique names, bail
            raise ConfigError("Not Enough Names in dictionary for number of nodes requested:%s/%s!"%(config.Nodes.count,len(naming_convention)))

        if bool(config.Nodes.node_names):
            # If given the correct number of names in the config file, do nothing
            if config.Nodes.count > len(config.Nodes.node_names):
                raise ConfigError("Not Enough Names in configfile for number of nodes requested!")
            assert int(config.Nodes.count) == len(config.Nodes.node_names)
        else:
            # Otherwise make up names from the naming_convention
            assert config.Nodes.count>=1, "No nodes configured"
            for n in range(config.Nodes.count):
                candidate_name= naming_convention[np.random.randint(0,len(naming_convention))]
                while candidate_name in [ x.name for x in self.nodes ]:
                    candidate_name= naming_convention[np.random.randint(0,len(naming_convention))]
                config.Nodes.node_names.append(candidate_name)
                self.logger.info("Gave node %d name %s"%(n,candidate_name))

        # LAYERCAKE CONFIG CHECK
        try:
            config.mac_mod=getattr(MAC,str(config.MAC.protocol))
        except AttributeError:
            raise ConfigError("Can't find MAC: %s"%config.MAC.protocol)

        try:
            config.net_mod=getattr(Net,str(config.Network.protocol))
        except AttributeError:
            raise ConfigError("Can't find Network: %s"%config.Network.protocol)

        try:
            config.app_mod=getattr(Applications,str(config.Application.protocol))
        except AttributeError:
            raise ConfigError("Can't find Application: %s"%config.Application.protocol)

        #Confirm
        self.logger.info("Built Config: %s"%str(config))
        return config

    def configureEnvironment(self, env_config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        return Environment(
            self,
            shape=env_config.shape,
            resolution=env_config.resolution,
            base_depth=env_config.base_depth
            )

    def configureNodes(self,node_config,fleet=None):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        nodelist = []
        #Configure specified nodes
        for nodeName in node_config.node_names:
            config=self.vectorGen(
                nodeName,
                node_config
            )
            self.logger.debug("Generating node %s with config %s"%(nodeName,config))
            new_node=Node(
                nodeName,
                self,
                config
            )
            nodelist.append(new_node)

        #TODO Fleet implementation

        return nodelist

    def vectorGen(self,nodeName,node_config):
        """
        If a node is named in the configuration file, use the defined initial vector
        otherwise, use configured behaviour to assign an initial vector
        """
        try:# If there is an entry, use it
            vector = node_config.initial_node_vectors[nodeName]
            self.logger.info("Gave node %s a configured initial vector: %s"%(nodeName,str(vector)))
        except KeyError:
            #TODO add additional behaviours, eg random within enviroment; distribution around a point, etc
            gen_style = node_config.vector_generation
            if gen_style == "random":
                vector = self.environment.random_position()
                self.logger.debug("Gave node %s a random vector: %s"%(nodeName,vector))
            elif gen_style == "center":
                vector= self.environment.position_around()
                self.logger.debug("Gave node %s a center vector: %s"%(nodeName,vector))
            else:
                vector = [0,0,0]
                self.logger.debug("Gave node %s a zero vector from %s"%(nodeName,gen_style))
        assert len(vector)==3, "Incorrectly sized vector"
        # BEHAVIOUR CONFIG CHECK
        try:
            behave_mod=getattr(Behaviour,str(node_config.Behaviour.protocol))
            self.logger.debug("Using behaviour: %s"%behave_mod)
        except AttributeError:
            raise ConfigError("Can't find Behaviour: %s"%node_config.Behaviour.protocol)

        config = dotdict({  'vector':vector,
                            'max_speed':node_config.max_speed,
                            'max_turn':node_config.max_turn,
                            'behave_mod':behave_mod,
                            'Behaviour':node_config.Behaviour
                        })
        return config

    def postProcess(self,log):
        """
        Performs output and data generation for a given simulation
        """
        fig = plt.figure()
        ax = axes3.Axes3D(fig)
        ax.set_xlim3d([0,self.environment.shape[0]])
        ax.set_ylim3d([0,self.environment.shape[1]])
        ax.set_zlim3d([0,self.environment.shape[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        for oid in nodeIDs(log):
            x=map(itemgetter(0),(map(itemgetter(1),objectLog(log,oid))))
            y=map(itemgetter(1),(map(itemgetter(1),objectLog(log,oid))))
            z=map(itemgetter(2),(map(itemgetter(1),objectLog(log,oid))))
            ax.plot(x,y,z,label=self.reverse_node_lookup(oid))

        plt.show()

    def deltaT(self,now,then):
        """
        Time in seconds between two simulation times
        """
        return (now-then)*self.config.Simulation.sim_interval


