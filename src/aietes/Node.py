from Layercake import Layercake
import Behaviour
import Applications
import numpy as np
import uuid

from aietes.Tools import *

class Node(Sim.Process):
	"""
	Generic Representation of a network node
	"""
	def __init__(self,name,simulation,node_config, vector=None):
		self.logger = baselogger.getChild("%s[%s]"%(self.__class__.__name__,name))
		self.id=uuid.uuid4() #Hopefully unique id
		Sim.Process.__init__(self,name=name)

		self.simulation=simulation
		self.config=node_config

		self.pos_log=np.empty((3,self.simulation.config.Simulation.sim_duration))
		self.vec_log=np.empty((3,self.simulation.config.Simulation.sim_duration))

		##############################
		# Physical Configuration
		##############################

		#Extract (X,Y,Z) vector from 6-vector as position
		assert len(vector) == 3, "Malformed Vector%s"%vector
		self.position=np.array(vector,dtype=np.float)
		#Implied six vector velocity
		self.velocity=np.array([0,0,0],dtype=np.float)
		self.highest_attained_speed =0.0

		self._lastupdate=Sim.now()

		##############################
		# Application and(or) Comms stack
		##############################
		try:
			app_mod=getattr(Applications,str(node_config['app']))
		except AttributeError:
			raise ConfigError("Can't find Application: %s"%node_config['app'])
		if app_mod.HAS_LAYERCAKE:
			self.layercake = Layercake(self,node_config)
		else:
			self.layercake = None
		self.app = app_mod(self,node_config['Application'], layercake=self.layercake)

		##############################
		#Propultion Capabilities
		##############################
		if isinstance(self.config['max_speed'],int):
			#Max speed is independent of direction
			self.max_speed=np.asarray([self.config['max_speed'],self.config['max_speed'], self.config['max_speed']])
		else:
			self.max_speed = np.asarray(self.config['max_speed'])
		assert len(self.max_speed) == 3

		if isinstance(self.config['max_turn'],int):
			#Max Turn Rate is independent of orientation
			self.max_turn=[self.config['max_turn'],self.config['max_turn'],self.config['max_turn']]
		else:
			self.max_turn = self.config['max_turn']
		assert len(self.max_turn) == 3

		##############################
		#Internal Configure Node Behaviour
		##############################
		try:
			behaviour = self.config['Behaviour']['protocol']
			behave_mod=getattr(Behaviour,str(behaviour))
		except AttributeError:
			raise ConfigError("Can't find Behaviour: %s"%behaviour)

		self.behaviour=behave_mod(node=self,bev_config=self.config['Behaviour'])

		##############################
		#Simulation Configuration
		self.internalEvent = Sim.SimEvent(self.name)
		##############################

		self.logger.debug('instance created')

	def activate(self):
		"""
		Fired on Sim Start
		"""
		Sim.activate(self,self.lifecycle())
		self.app.activate()
		if self.app.layercake:
			self.layercake.activate()

		#Tell the environment that we are here!
		self.simulation.environment.update(self.id,self.getPos())

	def assignFleet(self,fleet):
		"""
		Assign or Re-assign a node to a given Fleet object
		"""
		self.fleet=fleet

	def distanceTo(self,otherNode):
		assert hasattr(otherNode,position), "Other object has no position"
		assert len(otherNode.position)==3
		return distance(self.position,otherVector.position)

	def push(self,forceVector):
		assert len(forceVector==3), "Out of spec vector: %s,%s"%(forceVector,type(forceVector))
		new_velocity=np.array(self.velocity+forceVector,dtype=np.float)
		if any(abs(new_velocity)>self.max_speed):
			new_velocity/=mag(new_velocity)
			new_velocity*=mag(self.max_speed)
			if debug: self.logger.debug("Attempted Velocity: %s, clipped: %s"%(forceVector,new_velocity))
		else:
			if debug: self.logger.debug("Velocity: %s"%forceVector)
		self.velocity=new_velocity

	def wallCheck(self):
		"""
		Are we still in the bloody box?
		"""
		return (all(self.position<np.asarray(self.simulation.environment.shape)) and all(np.zeros(3)<self.position))

	def move(self):
		"""
		Update node status
		"""
		##############################
		#Positional information
		##############################
		old_pos = self.position.copy()
		dT = self.simulation.deltaT(Sim.now(),self._lastupdate)
		attempted_vector = np.array(self.velocity,dtype=np.float)*dT
		self.position+=attempted_vector
		if debug: self.logger.debug("Moving by %s at %s * %f from %s to %s"%(self.velocity,mag(self.velocity),dT,old_pos,self.position))
		if not self.wallCheck():
			self.logger.critical("WE'RE OUT OF THE ENVIRONMENT! %s, v=%s"%(self.position,attempted_vector))
			raise Exception("%s Crashed out of the environment at %s m/s"%(self.name,mag(attempted_vector)))
		self.pos_log[:,self._lastupdate]=self.position.copy()

		self.vec_log[:,self._lastupdate]=attempted_vector

		self.highest_attained_speed = max(self.highest_attained_speed,mag(attempted_vector))
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
			##############################
			#Update Node State
			##############################
			if debug: self.logger.debug('updating behaviour')
			self.behaviour.process()

			yield Sim.release, self,self.simulation.process_flag
			yield Sim.waituntil, self, self.simulation.inner_join
			yield Sim.request, self, self.simulation.move_flag

			##############################
			#Move Fleet
			##############################
			if debug: self.logger.debug('updating position, then waiting %s'%self.behaviour.update_rate)
			self.move()

			##############################
			#Update Fleet State
			##############################
			if debug: self.logger.debug('updating map')
			yield Sim.hold, self, self.behaviour.update_rate
			yield Sim.release, self, self.simulation.move_flag
			self.simulation.environment.update(self.id,self.getPos())

