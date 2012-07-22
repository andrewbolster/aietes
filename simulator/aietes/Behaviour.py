from SimPy import Simulation as Sim
import logging
import numpy as np
import scipy.spatial
from Tools import dotdict,map_entry,memory_entry,baselogger,distance
from operator import attrgetter,itemgetter


class Behaviour():
    """
    Generic Represnetation of a Nodes behavioural characteristics
    #TODO should be a state machine?
    """
    def __init__(self,node,bev_config):
        #TODO internal representation of the environment
        self.logger = node.logger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance from bev_config: %s'%bev_config)
        #self.logger.setLevel(logging.DEBUG)
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
        forceVector = self.responseVector(self.node.position,self.node.velocity)
        self.node.push(forceVector)
        return


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
        self.neighbour_min_rad = self.bev_config.neighbourhood_min_rad
        self.clumping_factor = self.bev_config.clumping_factor
        self.schooling_factor = self.bev_config.schooling_factor
        assert self.nearest_neighbours>0
        assert self.neighbourhood_max_rad>0
        assert self.neighbour_min_rad>0

    def _get_neighbours(self,position):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        """
        #Sort and filter Neighbours by distance
        neighbours_with_distance=[memory_entry(key,
                                               value.position,
                                               self.node.distance_to(value.position),
                                               self.simulation.reverse_node_lookup(key).name
                                              ) for key,value in self.neighbours.items()]
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
        forceVector= np.array([0,0,0],dtype=np.float)
        forceVector+= self.clumpingVector(position)
        forceVector+= self.repulsiveVector(position)
        forceVector+= self.localHeading(position)
        forceVector = self.avoidWall(position,velocity,forceVector)
        self.logger.debug("Response:%s"%(forceVector))
        return forceVector

    def avoidWall(self, position, velocity, forceVector):
        """
        Called by responseVector to avoid walls to a distance of half min distance
        """
        min_dist = self.neighbour_min_rad*2 
        avoid = False
        if any((np.zeros(3)+min_dist)>position):
            self.logger.error("Too Close to the Origin-surfaces: %s"%position)
            offending_dim = position.argmin()
            avoiding_position = position.copy()
            avoiding_position[offending_dim]=float(0.0)
            avoid = True
        elif any(position>(np.asarray(self.simulation.environment.shape) - min_dist)):
            self.logger.error("Too Close to the Upper-surfaces: %s"%position)
            offending_dim = position.argmax()
            avoiding_position = position.copy()
            avoiding_position[offending_dim]=float(self.simulation.environment.shape[offending_dim])
            avoid = True
        else:
            response = forceVector

        if avoid:
            response = 0.5 * (position-avoiding_position)
            #response = (avoiding_position-position)
            self.logger.error("Wall Avoidance:%s, Avoiding:%s,%s,%s"%(response,avoiding_position,position,offending_dim))

        return response



    def clumpingVector(self,position):
        """
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector=np.array([0,0,0],dtype=np.float)
        for neighbour in self._get_neighbours(self.node.position):
            vector+=np.array(neighbour.position)

        try:
            #This assumes that the map contains one entry for each non-self node
            neighbourhood_com=(vector)/min(self.nearest_neighbours,len(self.neighbours))
            self.logger.debug("Cluster Centre,position,factor,neighbours: %s,%s,%s,%s"%(neighbourhood_com,vector,self.clumping_factor,len(self.neighbours)))
            # Return the fudged, relative vector to the centre of the cluster
            forceVector= (neighbourhood_com-position)*self.clumping_factor
        except ZeroDivisionError:
            self.logger.error("Zero Division Error: Returning zero vector")
            forceVector= position-position

        self.logger.debug("Clump:%s"%(forceVector))
        return forceVector

    def repulsiveVector(self,position):
        """
        Repesents the Short Range Repulsion behaviour:
            If a node is too close, steer away from it
        """
        forceVector=np.array([0,0,0],dtype=np.float)
        for neighbour in self._get_neighbours(position):
            forceVector+=self.repulseFromPosition(position,neighbour.position)
        # Return an inverse vector to the obstacles
        self.logger.debug("Repulse:%s"%(forceVector))
        return forceVector

    def repulseFromPosition(self,position,repulsive_position):
        distanceVal=distance(position,repulsive_position)
        forceVector=2*(position-repulsive_position)/np.sqrt(distanceVal)
        assert distanceVal > 2, "Too close to %s"%(repulsive_position)
        return -forceVector


    def localHeading(self,position):
        """
        Represents Local Average Heading
        """
        vector=np.array([0,0,0])
        for neighbour in self.simulation.nodes:
            if neighbour is not self.node:
                vector+=neighbour.velocity
        forceVector = self.schooling_factor * vector / (len(self.simulation.nodes) - 1)
        self.logger.debug("Schooling:%s"%(forceVector))
        return forceVector

    def _percieved_vector(self,node_id):
        """
        Finite Difference Estimation
        from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
        """
        node_history=sorted(filter(lambda x: x.object_id==nodeid, self.memory), key=time)
        return (node_history[-1].position-node_history[-2].position)/(node_history[-1].time-node_history[-2].time)

