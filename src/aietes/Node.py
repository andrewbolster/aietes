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

	def __init__(self, name, simulation, node_config, vector = None):
		self.logger = baselogger.getChild("%s[%s]" % (self.__class__.__name__, name))
		self.id = uuid.uuid4() #Hopefully unique id
		Sim.Process.__init__(self, name = name)

		self.simulation = simulation
		self.config = node_config
		self.mass = 10#kg modeling remus 100

		# Positions Initialised to None to highlight mistakes; as Any position could be a bad position
		self.pos_log = np.empty((3, self.simulation.config.Simulation.sim_duration))
		self.pos_log.fill(None)

		# Vectors initialised to Zero as at least a Zero vector doesn't break things.
		self.vec_log = np.zeros((3, self.simulation.config.Simulation.sim_duration))
		# However, it does make it easier to debug the behave/move interplay....
		self.vec_log.fill(None)

		# Store contributions
		self.contributions_log = []



		##############################
		# Physical Configuration
		##############################

		#Extract (X,Y,Z) vector from 6-vector as position
		assert len(vector) == 3, "Malformed Vector%s" % vector
		self.position = np.array(vector, dtype = np.float)
		#Implied six vector velocity
		self.velocity = np.array([0, 0, 0], dtype = np.float)
		self.forceVector = np.array([0, 0, 0], dtype = np.float)

		self.highest_attained_speed = 0.0

		self._lastupdate = Sim.now()

		##############################
		# Application and(or) Comms stack
		##############################
		try:
			app_mod = getattr(Applications, str(node_config['app']))
		except AttributeError:
			raise ConfigError("Can't find Application: %s" % node_config['app'])
		if app_mod.HAS_LAYERCAKE:
			self.layercake = Layercake(self, node_config)
		else:
			self.layercake = None
		self.app = app_mod(self, node_config['Application'], layercake = self.layercake)

		##############################
		# Propultion Capabilities
		##############################
		if len(self.config['cruising_speed']) == 1:
			#cruising speed is independent of direction
			self.cruising_speed = np.asarray(
				[self.config['cruising_speed'][0], self.config['cruising_speed'][0], self.config['cruising_speed'][0]],
				dtype = np.float64)
		else:
			self.cruising_speed = np.asarray(self.config['cruising_speed'], dtype = np.float64)
		assert len(self.cruising_speed) == 3

		if len(self.config['max_speed']) == 1:
			#Max speed is independent of direction
			self.max_speed = np.asarray(
				[self.config['max_speed'][0], self.config['max_speed'][0], self.config['max_speed'][0]])
		else:
			self.max_speed = np.asarray(self.config['max_speed'])
		assert len(self.max_speed) == 3

		if len(self.config['max_speed']) == 1:
			#Max Turn Rate is independent of orientation
			self.max_turn = [self.config['max_turn'], self.config['max_turn'], self.config['max_turn']]
		else:
			self.max_turn = self.config['max_turn']
		assert len(self.max_turn) == 3

		##############################
		# Internal Configure Node Behaviour
		##############################
		try:
			behaviour = self.config['Behaviour']['protocol']
			behave_mod = getattr(Behaviour, str(behaviour))
		except AttributeError:
			raise ConfigError("Can't find Behaviour: %s" % behaviour)

		self.behaviour = behave_mod(node = self, bev_config = self.config['Behaviour'])

		##############################
		# Simulation Configuration
		self.internalEvent = Sim.SimEvent(self.name)
		##############################

		self.logger.debug('instance created')

	def activate(self):
		"""
		Fired on Sim Start
		"""
		Sim.activate(self, self.lifecycle())
		self.app.activate()
		if self.app.layercake:
			self.layercake.activate()

		#Tell the environment that we are here!
		self.simulation.environment.update(self.id, self.getPos(), self.getVec())

	def assignFleet(self, fleet):
		"""
		Assign or Re-assign a node to a given Fleet object
		"""
		self.fleet = fleet

	def wallCheck(self):
		"""
		Are we still in the bloody box?
		"""
		return (all(self.position < np.asarray(self.simulation.environment.shape)) and all(np.zeros(3) < self.position))

	def distanceTo(self, otherNode):
		assert hasattr(otherNode, "position"), "Other object has no position"
		assert len(otherNode.position) == 3
		return distance(self.position, otherNode.position)

	def push(self, forceVector, contributions = None):
		assert len(forceVector == 3) and not np.isnan(sum(forceVector)), "Out of spec vector: %s,%s" % (
		forceVector, type(forceVector))

		new_forceVector = np.array(self.velocity + forceVector / self.mass, dtype = np.float)
		if mag(new_forceVector) > any(self.cruising_speed):
			new_forceVector = self.cruiseControl(new_forceVector, self.velocity)
			if __debug__: self.logger.info("Normalized Velocity: %s, clipped: %s" % (forceVector, new_forceVector))
		else:
			if __debug__: self.logger.info("Velocity: %s" % forceVector)
		self.forceVector = new_forceVector

		self.contributions_log.append(contributions)


	def cruiseControl(self, velocity, prev_velocity):
		"""
		Attempt to maintain cruising velocity
		"""
		refactor = 1.0 / np.exp(-(mag(velocity) - max(self.cruising_speed)))
		new_V = (velocity / mag(velocity)) * refactor
		if debug:
			self.logger.info("Cruise: From %f against %f giving norm factor %f and vel of %f" % (
			mag(velocity), mag(self.cruising_speed), refactor, mag(new_V)))
		return new_V


	def move(self):
		"""
		Update node status
		"""
		##############################
		# Positional information
		##############################
		old_pos = self.position.copy()
		dT = self.simulation.deltaT(Sim.now(), self._lastupdate)
		self.velocity = np.array(self.forceVector, dtype = np.float) * dT
		self.position += self.velocity
		if debug: self.logger.debug("Moving by %s at %s * %f from %s to %s" % (
		self.velocity, mag(self.forceVector), dT, old_pos, self.position))
		if not self.wallCheck():
			self.logger.critical("Moving by %s at %s * %f from %s to %s" % (
			self.velocity, mag(self.forceVector), dT, old_pos, self.position))
			self.logger.critical("WE'RE OUT OF THE ENVIRONMENT! %s, v=%s" % (self.position, self.velocity))
			self.logger.critical("PosLog:(%d,%d)\n%s" % (
			self._lastupdate, 0, [mag(self.pos_log[:, x]) for x in range(self._lastupdate)]))
			raise Exception("%s Crashed out of the environment at %s m/s" % (self.name, mag(self.velocity)))
		self.pos_log[:, self._lastupdate] = self.position.copy()
		assert not np.isnan(sum(self.pos_log[:, self._lastupdate]))

		self.vec_log[:, self._lastupdate] = self.velocity

		self.highest_attained_speed = max(self.highest_attained_speed, mag(self.velocity))
		self._lastupdate = Sim.now()

	def setPos(self, placeVector):
		assert isinstance(placeVector, np.array)
		assert len(placeVector) == 3
		self.logger.info("Vector focibly moved")
		self.position = placeVector

	def getPos(self):
		return self.position.copy()

	def getVec(self):
		return self.velocity.copy()

	def distance_to(self, their_position):
		d = distance(self.getPos(), their_position)
		return d


	def lifecycle(self):
		"""
		Called to update internal awareness and motion:
			THESE CALLS ARE NOT GUARANTEED TO BE ALIGNED ACROSS NODES
		"""
		self.logger.info("Initialised Node Lifecycle")
		while(True):
			##############################
			#Update Node State
			##############################
			if debug: self.logger.info('updating behaviour')
			self.behaviour.process()

			yield Sim.passivate, self

			##############################
			#Move Fleet
			##############################
			if debug: self.logger.info('updating position, then waiting %s' % self.behaviour.update_rate)
			self.move()

			##############################
			#Update Fleet State
			##############################
			if debug: self.logger.info('updating map')
			yield Sim.hold, self, self.behaviour.update_rate
			self.simulation.environment.update(self.id, self.getPos(), self.getVec())

