__author__ = 'bolster'
import numpy as np
import logging, os
from scipy.spatial.distance import pdist, squareform
from aietes.Tools import mag

class DataPackage(object):
	"""
	Data Store for simulation results
	Replicates a Numpy Array as is directly accessible as an n-vector
	[x,y,z,t] array for n simulated nodes positions over time. Additional
	information is queriable through the object.
	"""

	def __init__(self, source = None,
	             p = None, v = None, names = None, environment = None, tmax = None,
	             *args, **kwargs):
		"""
		Raises IOError on load failure
		"""
		if source is not None:
			source_dataset = np.load(source)
			self.p = source_dataset['positions']
			self.v = source_dataset['vectors']
			self.names = source_dataset['names']
			try:
				self.contributions = source_dataset['contributions']
			except KeyError as e:
				logging.error("Using an old AIETES npz with no contributions")
				self.contributions = None
			self.environment = source_dataset['environment']
			try:
				self.title = getattr(source_dataset, 'title')
			except AttributeError:
				# If simulation title not explicitly given, use the filename -npz
				self.title = os.path.splitext(os.path.basename(source))[0]
		elif all(x is not None for x in [p, v, names, environment]):
			self.p = p
			self.v = v
			self.names = names
			self.environment = environment
			self.title = kwargs.get('title', "")
		else:
			raise ValueError("Can't work out what the hell you want!: %s" % str(kwargs))

		self.tmax = len(self.p[0][0]) if tmax is None else tmax
		self.n = len(self.p)

	def update(self, p = None, v = None, names = None, environment = None, **kwargs):
		logging.debug("Updating from tmax %d" % self.tmax)

		if all(x is not None for x in [p, v, names, environment]):
			self.p = p
			self.v = v
			self.names = names
			self.environment = environment
		else:
			raise ValueError("Can't work out what the hell you want!")
		self.tmax = len(self.p[0][0])
		self.n = len(self.p)

	#Data has the format:
	# [n][x,y,z][t]

	def position_of(self, node, time):
		"""
		Query the data set for the x,y,z position of a node at a given time
		"""
		try:
			position = self.p[node, :, time]
			if not np.isnan(sum(position)):
				return position
			else:
				raise IndexError("GAHHHH")

		except IndexError as e:
			logging.debug("Position Query for n:%d @ %d for position shape %s" % (node, time, self.p[node].shape))
			raise e

	def position_slice(self, time):
		"""
	Query the dataset for the [n][x,y,z] position list of all nodes at a given time
	"""
		return self.p[:, :, time]
	def position_slice_of(self, node):
		"""
	Query the dataset for the [n][x,y,z] position list of all nodes at a given time
	"""
		try:
			return self.p[node, :, :]
		except IndexError as e:
			logging.debug("Position Query for n:%d @ all for position shape %s" % (node, self.p[node].shape))
			raise e

	def heading_of(self, node, time):
		"""
		Query the data set for the x,y,z vector of a node at a given time
		"""
		return self.v[node, :, time]

	def heading_slice(self, time):
		"""
		Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
		"""
		return self.v[:, :, time]

	def heading_slice_of(self, node):
		"""
		Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
		"""
		try:
			return self.v[node, :, :]
		except IndexError as e:
			logging.debug("Heading Query for n:%d @ all for heading shape %s" % (node, self.v[node].shape))
			raise e

	def heading_mag_range(self):
		"""
		Returns an array of average heading magnitudes across the dataset
		i.e. the average speed for each node in the fleet at each time

		"""
		magnitudes = [sum(map(mag, self.heading_slice(time))) / self.n
		              for time in range(self.tmax)
		]
		return magnitudes

	def average_heading_mag_range(self):
		"""
		Returns an array of average heading magnitudes across the dataset
		i.e. the average speed for each node in the fleet at each time

		"""
		magnitudes = [mag(self.average_heading(time))
		              for time in range(self.tmax)
		]
		return magnitudes

	def heading_stddev_range(self):
		"""
		Returns an array of overall heading stddevs across the dataset

		"""
		deviations = [np.std(
			self.deviation_from_at(self.average_heading(time), time)
		)
		              for time in range(self.tmax)
		]
		return deviations

	def average_heading(self, time):
		"""
		Generate the average heading for the fleet at a given timeslice
		:param time: time index to calculate at
		:type int

		:raises ValueError
		"""
		if not (0 <= time <= self.tmax):
			raise ValueError("Time must be in the range of the dataset: %s, %s" % (time, self.tmax))

		return np.average(self.heading_slice(time), axis = 0)

	def deviation_from_at(self, heading, time):
		"""
		Return a one dimensional list of the linear deviation from
		each node to a given heading

		:param time: time index to calculate at
		:type int

		:param heading: (x,y,z) heading to compare against
		:type tuple
		"""
		return map(
			lambda v: np.linalg.norm(
				np.array(v) - np.array(heading)
			), self.heading_slice(time)
		)

	def average_position(self, time):
		"""
		Generate the average position (center) for the fleet at a given
		timeslice

		:param time: time index to calculate at
		:type int

		:raises ValueError
		"""
		if not (0 <= time <= self.tmax):
			raise ValueError("Time must be in the range of the dataset: %s, %s" % (time, self.tmax))

		return np.average(self.position_slice(time), axis = 0)

	def sphere_of_positions(self, time):
		"""
		Return the x,y,z,r configuration of the fleet encompassing
		it's location and size

		:param time: time index to calculate at
		:type int

		"""
		x, y, z = self.average_position(time)
		max_r = max(self.distances_from_at((x, y, z), time))

		return (x, y, z, max_r)

	def distances_from_at(self, position, time):
		"""
		Return a one dimensional list of the linear distances from
		each node to a given position

		:param time: time index to calculate at
		:type int

		:param position: (x,y,z) position to compare against
		:type tuple
		"""
		return map(
			lambda p: mag(
				np.array(p) - np.array(position)
			), self.position_slice(time)
		)

	def distances_from_average_at(self, time):
		"""
		Return a one dimensional list of the linear distances from
		each node to the current fleet average point
		"""
		return self.distances_from_at(self.average_position(time), time)


	def distance_from_average_stddev_range(self):
		"""
		Returns an array of overall stddevs across the dataset
		"""
		return [np.std(self.distances_from_average_at(time)) for time in range(self.tmax)]

	def position_matrix(self, time):
		"""
		Returns a positional matrix of the distances between all the nodes at a given time.
		"""
		return squareform(
			pdist(
				self.position_slice(time)
			)
		)

	def inter_distance_average(self, time):
		"""
		Returns the average distance between nodes
		"""
		return np.average(self.position_matrix(time))
