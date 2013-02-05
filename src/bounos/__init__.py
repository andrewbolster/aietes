# BOUNOS - Heir to the Kingdom of AIETES

import argparse

import numpy as np
from Plotting import *
import re

np.seterr(under = "ignore")

from Metrics import *
from Analyses import *
from DataPackage import DataPackage


class BounosModel(DataPackage):
	def __init__(self, *args, **kwargs):
		self.metrics = []
		self.is_ready = False
		self.is_simulating = None

	def import_datafile(self, file):
		super(BounosModel, self).__init__(source = file)
		self.is_ready = True
		self.is_simulating = False


	def update_data_from_sim(self, p, v, names, environment, now):
		"""
		Call back function used by SimulationStep if doing real time simulation
		Will 'export' DataPackage data from the running simulation up to the requested time (self.t)
		"""
		self.log.debug("Updating data from simulator at %d" % now)
		self.update(p = p, v = v, names = names, environment = environment)

	@classmethod
	def is_valid_aietes_datafile(self, file):
		test = re.compile("aietes.npz$")
		return test.search(file)


def main():
	"""
	Initial Entry Point; Does very little other that option parsing
	"""
	parser = argparse.ArgumentParser(description = "Simulation Visualisation and Analysis Suite for AIETES")
	parser.add_argument('--source', '-s',
	                    dest = 'source', action = 'store',nargs='+',
	                    metavar = 'XXX.npz',
	                    required = True,
	                    help = 'AIETES Simulation Data Package to be analysed'
	)

	args = parser.parse_args()
	print args
	if isinstance(args.source, list):
		sources = args.source
	else:
		sources = [args.source]
	for source in sources:
		data=DataPackage(source)
		#interactive_plot(data)
		metric = PerNode_Internode_Distance_Avg(data=data)
		metric.update()
		KF_metric_plot(metric)
