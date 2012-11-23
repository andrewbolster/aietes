import numpy as np
from aietes.Tools import  memory_entry,baselogger,distance,fudge_normal,debug
from operator import attrgetter


class Behaviour():
	"""
	Generic Represnetation of a Nodes behavioural characteristics
	#TODO should be a state machine?
	"""
	def __init__(self,*args,**kwargs):
		#TODO internal representation of the environment
		self.node=kwargs.get('node')
		self.bev_config=kwargs.get('bev_config')
		self.logger = self.node.logger.getChild("%s"%(self.__class__.__name__))
		self.logger.info('creating instance')
		if debug: self.logger.debug('from bev_config: %s'%self.bev_config)
		#self.logger.setLevel(logging.DEBUG)
		self.update_rate=1
		self.memory={}
		self.behaviours=[]
		self.simulation = self.node.simulation
		self.neighbours = {}

	def neighbour_map(self):
		"""
		Returns a filtered map pulled from the global environment excluding self
		"""
		orig_map=dict((k,v) for k,v in self.simulation.environment.map.items() if v.object_id != self.node.id)

		for k,v in orig_map.items():
			orig_map[k].position = fudge_normal(v.position,0.2)

		return orig_map

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
		self.nearest_neighbours = self.getNearestNeighbours(self.node.position,n_neighbours=self.n_nearest_neighbours)
		forceVector = self.responseVector(self.node.position,self.node.velocity)
		forceVector = fudge_normal(forceVector,0.012)
		self.node.push(forceVector)
		return

	def getNearestNeighbours(self,position,n_neighbours=None, distance=np.inf):
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
		if n_neighbours is not None:
			nearest_neighbours = nearest_neighbours[:n_neighbours]
		return nearest_neighbours

	def responseVector(self, position,velocity):
		"""
		Called on process: Returns desired vector
		"""
		forceVector= np.array([0,0,0],dtype=np.float)
		for behaviour in self.behaviours:
			forceVector += behaviour(position)
		forceVector = self.avoidWall(position,velocity,forceVector)
		if debug: self.logger.debug("Response:%s"%(forceVector))
		return forceVector

	def avoidWall(self, position, velocity, forceVector):
		"""
		Called by responseVector to avoid walls to a distance of half min distance
		"""
		response = np.zeros(shape=forceVector.shape)
		min_dist = self.neighbour_min_rad*2
		avoid = False
		if any((np.zeros(3)+min_dist)>position):
			if debug: self.logger.debug("Too Close to the Origin-surfaces: %s"%position)
			offending_dim = position.argmin()
			avoiding_position = position.copy()
			avoiding_position[offending_dim]=float(0.0)
			avoid = True
		elif any(position>(np.asarray(self.simulation.environment.shape) - min_dist)):
			if debug: self.logger.debug("Too Close to the Upper-surfaces: %s"%position)
			offending_dim = position.argmax()
			avoiding_position = position.copy()
			avoiding_position[offending_dim]=float(self.simulation.environment.shape[offending_dim])
			avoid = True
		else:
			response = forceVector

		if avoid:
			#response = 0.5 * (position-avoiding_position)
			response = self.repulseFromPosition(position,avoiding_position)
			#response = (avoiding_position-position)
			self.logger.error("Wall Avoidance:%s"%(response))

		return response




class Flock(Behaviour):
	"""
	Flocking Behaviour as modelled by three rules:
		Short Range Repulsion
		Local-Average heading
		Long Range Attraction
	"""
	def __init__(self,*args,**kwargs):
		Behaviour.__init__(self,*args,**kwargs)
		self.n_nearest_neighbours = self.bev_config.nearest_neighbours
		self.neighbourhood_max_rad = self.bev_config.neighbourhood_max_rad
		self.neighbour_min_rad = self.bev_config.neighbourhood_min_rad
		self.clumping_factor = self.bev_config.clumping_factor
		self.schooling_factor = self.bev_config.schooling_factor
		self.collision_avoidance_d = self.bev_config.collision_avoidance_d

		self.behaviours.append(self.clumpingVector)
		self.behaviours.append(self.avoidCollisionVector)
		self.behaviours.append(self.localHeading)

		assert self.n_nearest_neighbours>0
		assert self.neighbourhood_max_rad>0
		assert self.neighbour_min_rad>0

	def clumpingVector(self,position):
		"""
		Represents the Long Range Attraction factor:
			Head towards average fleet point
		"""
		vector=np.array([0,0,0],dtype=np.float)
		for neighbour in self.nearest_neighbours:
			vector+=np.array(neighbour.position)

		try:
			#This assumes that the map contains one entry for each non-self node
			self.neighbourhood_com=(vector)/len(self.nearest_neighbours)
			if debug: self.logger.debug("Cluster Centre,position,factor,neighbours: %s,%s,%s,%s"%(self.neighbourhood_com,vector,self.clumping_factor,len(self.nearest_neighbours)))
			# Return the fudged, relative vector to the centre of the cluster
			forceVector= (self.neighbourhood_com-position)*self.clumping_factor
		except ZeroDivisionError:
			self.logger.error("Zero Division Error: Returning zero vector")
			forceVector= position-position

		if debug: self.logger.debug("Clump:%s"%(forceVector))
		return forceVector

	def repulsiveVector(self,position):
		"""
		Repesents the Short Range Repulsion behaviour:
			Steer away from it based on a repulsive desire curve
		"""
		forceVector=np.array([0,0,0],dtype=np.float)
		for neighbour in self.nearest_neighbours:
			forceVector+=self.repulseFromPosition(position,neighbour.position)
		# Return an inverse vector to the obstacles
		if debug: self.logger.debug("Repulse:%s"%(forceVector))
		return forceVector * self.repulsive_factor

	def avoidCollisionVector(self,position):
		"""
		Repesents the Short Range Collision Avoidance behaviour:
			If a node is too close, steer away from it
		"""
		forceVector=np.array([0,0,0],dtype=np.float)
		for neighbour in self.nearest_neighbours:
			if distance(position,neighbour.position) < self.collision_avoidance_d:
				forceVector+=position-neighbour.position
			# Return an inverse vector to the obstacles
		if debug: self.logger.debug("Repulse:%s"%(forceVector))
		return forceVector

	def repulseFromPosition(self,position,repulsive_position):
		forceVector=np.array([0,0,0],dtype=np.float)
		distanceVal=distance(position,repulsive_position)
		forceVector=(position-repulsive_position)/float(distanceVal)
		assert distanceVal > 2, "Too close to %s"%(repulsive_position)
		if debug: self.logger.debug("Repulsion from %s: %s, at range of %s"%(forceVector, repulsive_position,distanceVal))
		return forceVector

	def attractToPosition(self,position,attractive_position):
		forceVector=np.array([0,0,0],dtype=np.float)
		distanceVal=distance(position,attractive_position)
		forceVector=(attractive_position-position)/float(distanceVal)
		if debug: self.logger.debug("Attraction to %s: %s, at range of %s"%(forceVector, attractive_position,distanceVal))
		return forceVector

	def localHeading(self,position):
		"""
		Represents Local Average Heading
		"""
		vector=np.array([0,0,0])
		for neighbour in self.simulation.nodes:
			if neighbour is not self.node:
				vector+=fudge_normal(neighbour.velocity,max(abs(neighbour.velocity))/3)
		forceVector = self.schooling_factor * vector / (len(self.simulation.nodes) - 1)
		if debug: self.logger.debug("Schooling:%s"%(forceVector))
		return forceVector

	def _percieved_vector(self,node_id):
		"""
		Finite Difference Estimation
		from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
		"""
		node_history=sorted(filter(lambda x: x.object_id==nodeid, self.memory), key=time)
		return (node_history[-1].position-node_history[-2].position)/(node_history[-1].time-node_history[-2].time)


class Waypoint(Flock):
	def __init__(self,*args,**kwargs):
		Flock.__init__(self,*args,**kwargs)
		self.waypoint_factor = self.bev_config.waypoint_factor
		if not hasattr(self,str(self.bev_config.waypoint_style)):
			raise ValueError("Cannot generate using waypoint definition:%s"%self.bev_config.waypoint_style)
		else:
			generator=attrgetter(str(self.bev_config.waypoint_style))
			g=generator(self)
			self.logger.info("Generating waypoints: %s"%g.__name__)
			g()
		self.behaviours.append(self.waypointVector)

	def patrolCube(self):
		"""
		Generates a cubic patrol loop within the environment
		"""
		shape=np.asarray(self.simulation.environment.shape)
		prox=50
		cubedef=np.asarray(
			[[0,0,0],[1,0,0],[1,1,0],[0,1,0],
			 [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
		)
		self.cubepatrolroute=[(shape*(((vertex-0.5)/3)+0.5),prox) for vertex in cubedef]
		self.nextwaypoint=waypoint(self.cubepatrolroute)
		self.nextwaypoint.makeLoop(self.nextwaypoint)


	def waypointVector(self,position):
		forceVector=np.array([0,0,0],dtype=np.float)
		if self.nextwaypoint is not None:
			if distance(self.neighbourhood_com,self.nextwaypoint.position)<self.nextwaypoint.prox:
				self.logger.info("Moving to Next waypoint:%s"%self.nextwaypoint.position)
				self.nextwaypoint=self.nextwaypoint.next
			forceVector=self.attractToPosition(position, self.nextwaypoint.position)
		return forceVector*0.1


class waypoint(object):
	def __init__(self,positions):
		"""
		Defines waypoint paths:
			positions = [ [ position, proximity ], *]
		"""
		self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
		(self.position,self.prox)=positions[0]
		self.logger.info("Waypoint: %s,%s"%(self.position,self.prox))
		if len(positions) == 1:
			self.logger.info("End of Position List")
			self.next=None
		else:
			self.next=waypoint(positions[1:])

	def append(self,position):
		if self.next is None:
			self.next=waypoint([position])
		else:
			self.next.append(position)

	def insert(self,position):
		temp_waypoint = self.next
		self.next = waypoint([position])
		self.next.next=temp_waypoint

	def makeLoop(self,head):
		if self.next is None:
			self.next = head
		else:
			self.next.makeLoop(head)







