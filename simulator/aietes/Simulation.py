import SimPy.Simulation as Sim
import logging
import numpy
import scipy
import Layercake
import environment

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
            self.config = self.validateConfig(config_file)
            self.environment = configureEnvironment()
            self.nodes = configureNodes()
        except ConfigError as err:
            self.logger.err("Error in configuration, cannot continue: %s"%err)

    def configureEnvironment(self, host, config=self.config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        self.environment=Environment(shape=config.shape,
                                     scale=config.scale,
                                     base_depth=config.base_depth
                                    )
        pass

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

class VectorConfiguration():
    """
    Class that implements 6-degrees-of-freedom position/orientation tuples and their basic operations
    Convention is to use X,Y,Z,yaw,pitch,roll
    Angles in radians and with respect to global positional plane reference
    Angles denote the orientation of the given vector
    """
    self.position = numpy.array([0,0,0])
    self.orientation = numpy.array([0,0,0])
    def __init__(self,seq=[0,0,0,0,0,0])
        assert len(seq) == 6
        #Extract (X,Y,Z) vector from 6-vector as position
        self.position=seq[:3]
        #Extract (A,B,G) vector from 6-vector as orientation
        self.orientation=seq[3:]

    def distanceTo(self,otherVector):
        assert isinstance(otherVector, VectorConfiguration)
        return scipy.spatial.distance.euclidean(self.position,otherVector.position)

    def angleWith(self, otherVector):
        #TODO This doesn't work if any vector value is zero!
        assert isinstance(otherVector, VectorConfiguration)
        dot = numpy.dot(self.orientation, otherVector.orientation)
        c = dot / numpy.norm(self.orientation) / numpy.norm(otherVector.orientation)
        return numpy.arccos(c)

    def move(self,forceVector):
        assert isinstance(forceVector, VectorConfiguration)
        self.position+=forceVector.position
        self.orientation+=forceVector.orientation

    def setPos(self,placeVector):
        assert isinstance(placeVector,numpy.array)
        assert len(placeVector) == 3
        self.logger.info("Vector focibly moved")
        self.position = placeVector


class Node(Sim.process,VectorConfiguration):
    """
    Generic Representation of a network node
    """
    def __init__(self,name,simulation,vector=None,fleet=None):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))o
        self.logger.info('creating instance')
        #TODO Add auto-naming
        Sim.Process.__init__(self,name=name)
        self.simulation=simulation
        self.fleet=fleet
        # Physical Configuration
        VectorConfiguration.__init__(self,seq=vector)
        # Comms Stack
        self.layers = LayerCake(self,simulation)

    def lifecycle(self):
        """
        Called to update internal awareness and motion
        """
        while True:

            yield








