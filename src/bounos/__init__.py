# BOUNOS - Heir to the Kingdom of AIETES

import sys
import os
import traceback
import argparse

import numpy as np
import scipy as sp



from Plotting import interactive_plot

class DataPackage():
	"""
	Data Store for simulation results
	Replicates a Numpy Array as is directly accessible as an n-vector
	[x,y,z,t] array for n simulated nodes positions over time. Additional
	information is queriable through the object.
	"""

	def __init__(self,source,*args,**kwargs):
		#TODO Try Except this to be safe
		source_dataset = np.load(source)
		self.p = source_dataset['positions']
		self.v = source_dataset['vectors']
		self.names = source_dataset['names']
		self.environment = source_dataset['environment']
		self.tmax = len(self.p[0][0])
		self.n = len(self.p)
		try:
			self.title = getattr(source_dataset,'title')
		except AttributeError:
			# If simulation title not explicitly given, use the filename -npz
			self.title = os.path.splitext(os.path.basename(source))[0]

		#Data has the format:
			# [n][x,y,z][t]

	def position_of(self,node,time):
		"""
		Query the data set for the x,y,z position of a node at a given time
		"""
		return [self.p[node][dimension][time] for dimension in 0,1,2]

	def position_slice(self,time):
		"""
	Query the dataset for the [n][x,y,z] position list of all nodes at a given time
	"""
		return [ self.position_of(x,time) for x in range(self.n) ]

	def heading_of(self,node,time):
		"""
		Query the data set for the x,y,z vector of a node at a given time
		"""
		return [self.v[node][dimension][time] for dimension in 0,1,2]

	def heading_slice(self,time):
		"""
		Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
		"""
		return [ self.heading_of(x,time) for x in range(self.n) ]


	def trail_of(self,node,time=None):
		"""
		Return the [X:][Y:][Z:] trail for a given node from sim_start to
		a particular time

		If no time given, assume the full time range
		"""
		if time is None:
			time = self.tmax

		return [self.p[node][dimension][0:time] for dimension in 0,1,2]

	def average_heading(self, time):
		"""
		Generate the average heading for the fleet at a given timeslice
		:param time: time index to calculate at
		:type int

		:raises ValueError
		"""
		if not (0 <= time <= self.tmax):
			raise ValueError("Time must be in the range of the dataset")

		return sum(self.heading_slice(time))/float(self.n)

	def average_position(self, time):
		"""
		Generate the average position (center) for the fleet at a given
		timeslice

		:param time: time index to calculate at
		:type int

		:raises ValueError
		"""
		if not (0 <= time <= self.tmax):
			raise ValueError("Time must be in the range of the dataset")

		average = np.zeros(3,dtype=np.float)
		for element in self.position_slice(time):
			average += element

		return average/float(self.n)

	def sphere_of_positions(self, time):
		"""
		Return the x,y,z,r configuration of the fleet encompassing
		it's location and size

		:param time: time index to calculate at
		:type int

		"""
		x,y,z = self.average_position(time)
		max_r = max(self.distances_from_at((x,y,z),time))

		return (x,y,z,max_r)

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
			lambda p: np.linalg.norm(
				np.array(p)-np.array(position)
			),self.position_slice(time)
		)

	def sphere_of_positions_with_stddev(self, time):
		"""
		Return the x,y,z,r configuration of the fleet encompassing
		it's location and size

		:param time: time index to calculate at
		:type int

		"""
		average = self.average_position(time)
		distances = self.distances_from_at(average,time)
		r = max(distances)

		return average, r, np.std(distances)

	def stddev_range(self):
		"""
		Returns an array of overall stddevs across the dataset
		"""
		return [ self.sphere_of_positions_with_stddev(time)[-1] for time in range(self.tmax)]


def main():
	"""
	Initial Entry Point; Does very little other that option parsing
	"""
	parser = argparse.ArgumentParser(description="Simulation Visualisation and Analysis Suite for AIETES")
	parser.add_argument('--source','-s',
						dest='source', action='store',
						metavar='XXX.npz',
						required=True,
						help='AIETES Simulation Data Package to be analysed'
					   )


	args=parser.parse_args()
	interactive_plot(DataPackage(args.source))


