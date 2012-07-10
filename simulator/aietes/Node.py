import SimPy.Simulation as Sim
from Layercake import Layercake
import logging
import numpy as np
import uuid
from operator import attrgetter,itemgetter

from Tools import baselogger,distance

class Node(Sim.Process):
    """
    Generic Representation of a network node
    """
    def __init__(self,name,simulation,node_config):
        self.logger = baselogger.getChild("%s[%s]"%(self.__class__.__name__,name))
        self.logger.info('creating instance')
        self.id=uuid.uuid4() #Hopefully unique id

        Sim.Process.__init__(self,name=name)

        self.simulation=simulation
        self.config=node_config

        self.pos_log=[]

        # Physical Configuration

        #Extract (X,Y,Z) vector from 6-vector as position
        assert len(self.config.vector) == 3, "Malformed Vector%s"%self.config.vector
        self.position=np.array(self.config.vector)
        #Implied six vector velocity
        self.velocity=np.array([0,0,0])

        self._lastupdate=Sim.now()

        # Comms Stack
        self.layercake = Layercake(self,simulation)

        #Propultion Capabilities
        if isinstance(self.config.max_speed,int):
            #Max speed is independent of direction
            self.max_speed=[self.config.max_speed,self.config.max_speed, self.config.max_speed]
        else:
            self.max_speed = self.config.max_speed
        assert len(self.max_speed) == 3

        if isinstance(self.config.max_turn,int):
            #Max Turn Rate is independent of orientation
            self.max_turn=[self.config.max_turn,self.config.max_turn,self.config.max_turn]
        else:
            self.max_turn = self.config.max_turn
        assert len(self.max_turn) == 3

        #Internal Configure Node Behaviour
        self.behaviour=self.config.behave_mod(self,self.config.Behaviour)

        #Simulation Configuration
        self.internalEvent = Sim.SimEvent(self.name)

        self.logger.info('instance created')

    def activate(self):
        """
        Fired on Sim Start
        """
        Sim.activate(self,self.lifecycle())
        self.layercake.activate()

        #Tell the environment that we are here!
        self.simulation.environment.update(self.id,self.getPos())



    def distanceTo(self,otherNode):
        assert hasattr(otherNode,position), "Other object has no position"
        assert len(otherNode.position)==3
        return distance(self.position,otherVector.position)

    def push(self,forceVector):
        assert len(forceVector==3)
        self.velocity=np.array(forceVector,dtype=np.float)

    def move(self):
        """
        Update node status
        """
        #Positional information
        old_pos = self.position.copy()
        self.position +=np.array(self.velocity*(Sim.now()-self._lastupdate))
        self.logger.debug("Moving by %s * %s from %s to %s"%(self.velocity,(Sim.now()-self._lastupdate),old_pos,self.position))
        self.pos_log.append((self.position.copy(),self._lastupdate))
        self._lastupdate = Sim.now()

    def setPos(self,placeVector):
        assert isinstance(placeVector,np.array)
        assert len(placeVector) == 3
        self.logger.info("Vector focibly moved")
        self.position = placeVector

    def getPos(self):
        return self.position.copy()

    def distance_to(self, their_position):
        d = distance(self.getPos(),their_position)
        return d


    def lifecycle(self):
        """
        Called to update internal awareness and motion:
            THESE CALLS ARE NOT GUARANTEED TO BE ALIGNED ACROSS NODES
        """
        self.logger.info("Initialised Node Lifecycle")
        while(True):
            yield Sim.request, self,self.simulation.process_flag
            yield Sim.waituntil, self, self.simulation.outer_join
            #Update Node State
            self.logger.debug('updating behaviour')
            self.behaviour.process()

            yield Sim.release, self,self.simulation.process_flag
            yield Sim.waituntil, self, self.simulation.inner_join
            yield Sim.request, self, self.simulation.move_flag
            #Move Fleet
            self.logger.debug('updating position, then waiting %s'%self.behaviour.update_rate)
            self.move()

            #Update Fleet State
            self.logger.debug('updating map')
            self.simulation.environment.update(self.id,self.getPos())
            yield Sim.hold, self, self.behaviour.update_rate
            yield Sim.release, self, self.simulation.move_flag

