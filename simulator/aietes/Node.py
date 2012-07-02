import SimPy.Simulation as Sim
import Layercake
import logging, numpy
from operator import attrgetter,itemgetter

module_logger = logging.getLogger('AIETES.Node')
class VectorConfiguration():
    """
    Class that implements Node Positional and velocity information
    Convention is to use X,Y,Z
    Angles in radians and with respect to global positional plane reference
    """
    #TODO Expand this to proper 6d vector (yaw,pitch,roll
    def __init__(self,seq=[0,0,0]):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')
        assert len(seq) == 3
        #Extract (X,Y,Z) vector from 6-vector as position
        self.position=numpy.array(seq)
        #Implied six vector velocity
        self.velocity=numpy.array([0,0,0])

        self._lastupdate=Sim.now()

    def distanceTo(self,otherVector):
        assert isinstance(otherVector, VectorConfiguration)
        return scipy.spatial.distance.euclidean(self.position,otherVector.position)

    def push(self,forceVector):
        assert len(forceVector==3)
        self.velocity+=numpy.array(forceVector)

    def _update(self):
        """
        Update position
        """
        self.position +=numpy.array(self.velocity*(self._lastupdate-Sim.now()))
        self._lastupdate = Sim.now()

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
        self.node=node
        self.config=config
        self.map=node.map
        self.memory={}
        self._init_behaviour()

    class memory_entry():
        def __init__(self,object_id,position):
            self.object_id=object_id
            self.position=position
            self.time=Sim.now()

    class map_entry():
        def __init__(self,node):
            self.position=node.position
            self.velocity=node.velocity
            self.time=Sim.now()

    def _init_behaviour():
        pass

    def update():
        self.map[self.node.id]=self.map_entry(self.node)
        pass

    def move():
        pass

    def addMemory(object_id,position):
        """
        Called by node lifecycle to update the internal representation of the environment
        """
        #TODO expand this to do SLAM?
        self.memory+=memory_entry(object_id,position)

    def responseVector():
        """
        Returns a 6-force-vector indicating the direction / orientation in which to move
        """
        forceVector= numpy.array([0,0,0])
        return VectorConfiguration()

    def distance(self,my_position, their_position):
        return scipy.spatial.distance.euclidean(my_position,their_position)

class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """
    def _init_behaviour(self):
        self.nearest_neighbours = config.nearest_neighbours
        self.neighbourhood_max_rad = config.neighbourhood_max_rad
        self.neighbourhood_max_dt= config.neighbourhood_max_dt
        self.neighbour_min_rad = config.neighbour_min_rad

    def _get_neighbours(self,position):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        """
        #Sort and filter Neighbours by distance
        neighours=filter(lambda x:x[0]<=self.neighbourhood_max_rad
        ,sorted(
            map(
                None,
                map(
                    lambda x: self.distance(position,x.position),
                    self.map
                    )
                ,self.map
                )
            ,key=itemgetter(0)
            )
        )
        #Select N neighbours in order
        return neighbours[:self.nearest_neighbours]

    def responseVector(self,position,velocity):
        """
        Called on process: Returns desired vector
        """
        forceVector= numpy.array([0,0,0])
        forceVector+= self.clumpingVector(position)
        forceVector+= self.replusiveVector(position)
        forceVector+= self.localHeading(position)

        return forceVector

    def clumpingVector(self,position):
        """
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector=numpy.array([0,0,0])
        for neighbour in neighbours:
            vector+=numpy.array(neighbour.position)

        #This assumes that the map contains one entry for each non-self node
        neighbourhood_com=vector/min(self.nearest_neighbours,len(self.map))

        # Return the fudged, relative vector to the centre of the cluster
        return (neighbourhood_com-position)/self.config.clumping_factor

    def replusiveVector(self,position):
        """
        Repesents the Short Range Repulsion behaviour:
            If a node is too close, steer away from it
        """
        #TODO Test if this is better as a scalar function rather than a step value

        vector=numpy.array([0,0,0])
        for neighbour in self._get_neighbours(position):
            if distance(position,neighbour.position) > self.neighbour_min_rad:
                #Too Close, Move away
                vector-=(position-neighbour.position)

        # Return an inverse vector to the obstacles
        return vector

    def localHeading(self,velocity):
        """
        Represents Local Average Heading
        """
        vector=numpy.array([0,0,0])
        for neighbour in self._get_neighbours(position):
            vector += neighbour.p_velocity
        return vector

    def _percieved_vector(self,node_id):
        """
        Finite Difference Estimation
        from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
        """
        node_history=sorted(filter(lambda x: x.object_id==nodeid, self.memory), key=time)
        return (node_history[-1].position-node_history[-2].position)/(node_history[-1].time-node_history[-2].time)

class Node(Sim.Process,VectorConfiguration):
    """
    Generic Representation of a network node
    """
    def __init__(self,name,simulation,config):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')
        self.id=uuid.uuid4() #Hopefully unique id
        #TODO Add auto-naming
        Sim.Process.__init__(self,name=name)
        self.simulation=simulation
        self.config=config
        # Physical Configuration
        VectorConfiguration.__init__(self,seq=config.initial_vector)
        # Comms Stack
        self.layers = Layercake(self,simulation)

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
        Called to update internal awareness and motion:
            THESE CALLS ARE NOT GUARANTEED TO BE ALIGNED ACROSS NODES
        """
        while(True):
            #Update Fleet State
            yield Sim.request, self, self.simulation.update_flag
            self.behaviour.update()
            yield Sim.release, self, self.simulation.update_flag
            yield Sim.waituntil, self, self.simulation.clearToStep()
            #Update Node State
            yield Sim.request, self, self.simulation.process_flag
            self.behaviour.process()
            yield Sim.release, self, self.simulation.process_flag
            yield Sim.waituntil, self, self.simulation.clearToStep()
            #Move Fleet
            yield Sim.request, self, self.simulation.move_flag
            self.behaviour.move()
            yield Sim.release, self, self.simulation.move_flag
            yield Sim.waituntil, self, self.simulation.clearToStep()
            yield Sim.hold, self, self.behaviour.update_rate


