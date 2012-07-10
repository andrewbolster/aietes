from SimPy import Simulation as Sim
import logging
from Tools import baselogger,dotdict,map_entry
from collections import namedtuple
import numpy as np
import uuid
Log = namedtuple('Log',['name','object_id','time','position'], verbose=True)
class Environment():
    """
    Environment Class representing the physical environment inc any objects
    / activities within that environment that are not controlled by the
    simulated entities i.e. wind, tides, speed of sound at depth, etc
    """
    def __init__(self,simulation,shape=[100,100,100],resolution=1,base_depth=-1000,sos_model=None):
        """
        Generate a box with points from 0 to (size) in each dimension, where 
        each point represents a cube of side resolution metres:
            Volume is the representation of the physical environment (XYZ)
            Map is the 
        """
        self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance')
        self.volume=np.ndarray(shape=shape,dtype=uuid.UUID)
        self.map={}
        self.pos_log=[]
        self.depth=base_depth
        self.sos=1400
        self.simulation = simulation
        #TODO Random Surface Generation
        #self.generateSurface()
        #TODO 'Tidal motion' factor


    def random_position(self,want_empty=True):
        """
        Return a random empty map reference within the environment volume
        """
        is_empty=False
        while not is_empty:
            ran_x = np.random.randint(0,self.volume.shape[0])
            ran_y = np.random.randint(0,self.volume.shape[1])
            ran_z = np.random.randint(0,self.volume.shape[2])
            is_empty = not (want_empty or bool(self.volume[ran_x, ran_y, ran_z ]))

        return [ran_x, ran_y, ran_z ]

    def position_around(self,position=None):
        """
        Return a nearly-random map entry within the environment volume around a given position
        """
        if position is None:
            position = np.asarray(self.volume.shape)/2
        if self.is_outside(position):
            raise ValueError("Position is not within volume")
        else:
            valid = False
            while not valid:
                candidate_pos=np.random.normal(np.asarray(position),5)
                candidate_pos = tuple(np.asarray(candidate_pos,dtype=int))
                valid = self.volume[candidate_pos] is None
                self.logger.debug("Candidate position: %s:%s"%((candidate_pos),valid))
        return candidate_pos


    def is_outside(self,position):
        too_high = not all(position<self.volume.shape)
        too_low = not all(position>0)
        return too_high or too_low


    def update(self,object_id,position):
        """
        Update the environment to reflect a movement
        """
        object_name = self.simulation.reverse_node_lookup(object_id).name
        try:
            self.logger.info("Moving %s from %s to %s"%(object_name,
                                                         self.map[object_id].position,
                                                         position)
                             )
            self.map[object_id]=map_entry(object_id,position,object_name)
        except KeyError:
            self.logger.debug("Creating map entry for %s at %s"%(object_name,position))
            self.map[object_id]=map_entry(object_id,position,object_name)
        self.volume[tuple(position)]=object_id
        self.pos_log.append(Log(name=object_name,
                                position=position,
                                object_id=object_id,
                                time=Sim.now()
                               ))


    def export(self,filename=None):
        """
        Export the current environment to a csv
        """
        assert filename is not None
        np.savez(filename, self.pos_log)


