from SimPy import Simulation as Sim
import logging
import numpy
import scipy.spatial
from Tools import dotdict,map_entry,memory_entry
from operator import attrgetter,itemgetter


class Behaviour():
    """
    Generic Represnetation of a Nodes behavioural characteristics
    #TODO should be a state machine?
    """
    def __init__(self,node,bev_config):
        #TODO internal representation of the environment
        self.logger = logging.getLogger("%s.%s"%(node.logger.name,self.__class__.__name__))
        self.logger.info('creating instance from bev_config: %s'%bev_config)
        self.node=node
        self.bev_config=bev_config
        self.update_rate=1
        self.memory={}
        self._init_behaviour()
        self.simulation = self.node.simulation
        self.neighbours = {}



    def neighbour_map(self):
        """
        Returns a filtered map pulled from the global environment excluding self
        """
        return dict((k,v) for k,v in self.simulation.environment.map.items() if v.object_id != self.node.id)

    def _init_behaviour(self):
        pass

    def update(self):
        """
        Update local (and global) knowledge with the current state of the object
        """
        self.simulation.environment.update(self.node.id,self.node.position)
        pass

    def move(self):
        pass

    def addMemory(self,object_id,position):
        """
        Called by node lifecycle to update the internal representation of the environment
        """
        #TODO expand this to do SLAM?
        self.memory+=memory_entry(object_id,position)

    def process(self):
        """
        Process current map and memory information and update velocities
        """
        self.neighbours = self.neighbour_map()
        return self.responseVector(self.node.position,self.node.velocity)

    def distance(self, their_position):
        return numpy.linalg.norm(self.node.position - their_position)

class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """
    def _init_behaviour(self):
        self.nearest_neighbours = self.bev_config.nearest_neighbours
        self.neighbourhood_max_rad = self.bev_config.neighbourhood_max_rad
        self.neighbour_min_rad = self.bev_config.neighbour_min_rad
        self.clumping_factor = self.bev_config.clumping_factor

    def _get_neighbours(self,position):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        """
        #Sort and filter Neighbours by distance
        neighbours_with_distance=[memory_entry(key,value.position,self.distance(value.position)) for key,value in self.neighbours.items()]
        #self.logger.debug("Got Distances: %s"%neighbours_with_distance)
        nearest_neighbours=sorted(neighbours_with_distance
            ,key=attrgetter('distance')
            )
        #Select N neighbours in order
        #self.logger.debug("Nearest Neighbours:%s"%nearest_neighbours)
        return nearest_neighbours

    def responseVector(self,position,velocity):
        """
        Called on process: Returns desired vector
        """
        forceVector= numpy.array([0,0,0])
        forceVector+= self.clumpingVector(position)
        forceVector+= self.replusiveVector(position)
        #forceVector+= self.localHeading(position)
        self.logger.debug("%s:%s"%(__name__,forceVector))
        return forceVector

    def clumpingVector(self,position):
        """
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector=numpy.array([0,0,0])
        for neighbour in self._get_neighbours(self.node.position):
            vector+=numpy.array(neighbour.position)

        try:
            #This assumes that the map contains one entry for each non-self node
            neighbourhood_com=vector/min(self.nearest_neighbours,len(self.neighbour_map()))
            # Return the fudged, relative vector to the centre of the cluster
            forceVector= (neighbourhood_com-position)/self.clumping_factor
        except ZeroDivisionException:
            forceVector= position-position
        self.logger.debug("%s:%s"%(__name__,forceVector))
        return forceVector

    def replusiveVector(self,position):
        """
        Repesents the Short Range Repulsion behaviour:
            If a node is too close, steer away from it
        """
        #TODO Test if this is better as a scalar function rather than a step value

        forceVector=numpy.array([0,0,0])
        for neighbour in self._get_neighbours(position):
            if self.distance(neighbour.position) > self.neighbour_min_rad:
                #Too Close, Move away
                forceVector-=(position-neighbour.position)

        # Return an inverse vector to the obstacles
        self.logger.debug("%s:%s"%(__name__,forceVector))
        return forceVector

    def localHeading(self,position):
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

