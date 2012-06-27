import SimPy.Simulation as Sim
import LayerCake
import logging

module_logger = logging.getLogger('AIETES.Node')
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
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))o
        self.logger.info('creating instance')
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

class Behaviour():
    """
    Generic Represnetation of a Nodes behavioural characteristics
    #TODO should be a state machine?
    """
    def __init__(self,node,config):
        #TODO internal representation of the environment
        self.node
        self.config
        self.memory=[]
        self._init_behaviour()

    def _init_behaviour():
        pass

    def updateMap(information):
        """
        Called by node lifecycle to update the internal representation of the environment
        """
        pass

    def responseVector():
        """
        Returns a 6-force-vector indicating the direction / orientation in which to move
        """
        return VectorConfiguration()


class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """
    def _init_behaviour():
        self.nearest_neighbours = config.nearest_neighbours
        self.neighbourhood_max_rad = config.neighbourhood_max_rad

    def responseVector():
        """
        Returns a 6-force-vector indicating the direction / orientation in which to move
        """
        cohesionpoint=numpy.average([v.position for v in self.map])
        #TODO Add Weight based on time last seen




class Node(Sim.process,VectorConfiguration):
    """
    Generic Representation of a network node
    """
    def __init__(self,name,simulation,config):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))o
        self.logger.info('creating instance')
        #TODO Add auto-naming
        Sim.Process.__init__(self,name=name)
        self.simulation=simulation
        self.config=config
        # Physical Configuration
        VectorConfiguration.__init__(self,seq=config.initial_vector)
        # Comms Stack
        self.layers = LayerCake(self,simulation)

        #Propultion Capabilities
        if isinstance(config.max_speed,int):
            #Max speed is independent of direction
            self.max_speed=[config.max_speed,config.max_speed, config.max_speed]
        else:
            self.max_speed = config.max_speed
        assert len(self.max_speed) == 3

        if isinstance(config.max_turn,int):
            #Max Turn Rate is independent of orientation
            self.max_turn=[config.max_turn,config.max_turn,config.max_turn]
        else:
            self.max_turn = config.max_turn
        assert len(self.max_turn) == 3





        #Internal Representation of the environment
        behaviour_type=getattr(self.config.behaviour_type)
        self.behaviour=behaviour_type(self.config.behaviour)


    def lifecycle(self):
        """
        Called to update internal awareness and motion
        """
        while(True):
            self.behaviour.updateMap()
            self.behaviour.move()

            yield


