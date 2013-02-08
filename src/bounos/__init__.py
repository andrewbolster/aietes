# BOUNOS - Heir to the Kingdom of AIETES

import argparse

import numpy as np
from Plotting import *
import re

np.seterr(under = "ignore")

from Metrics import *
from Analyses import *
from DataPackage import DataPackage
from XKCDify import XKCDify

from aietes.Tools import list_functions, itersubclasses
font = {'family' : 'normal',
		'weight' : 'normal',
		'size'	 : 10}


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
		#TODO This isn't a very good test...
		test = re.compile(".npz$")
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
	parser.add_argument('--output', '-o',
						dest = 'output', action = 'store',
						default = None,
						metavar = 'png|pdf',
						help = 'Output to png/pdf'
	)
	parser.add_argument('--title', '-T',
						dest = 'title', action = 'store',
						default = "bounos_figure",
						metavar = '<Filename>',
						help = 'Set a title for this analysis run'
	)
	parser.add_argument('--outdim', '-d',
						dest = 'dims', action = 'store', nargs=2, type=int,
						default = None,
						metavar = '2.3',
						help = 'Figure Dimensions in Inches (default autofit)'
	)
	parser.add_argument('--comparison', '-c', dest = 'compare',
						action = 'store_true', default = False,
						help = "Compare Two Datasets for Meta-Analysis"
	)
	parser.add_argument('--xkcdify', '-x', dest = 'xkcd',
						action = 'store_true', default = False,
						help = "Plot like Randall"
	)
	parser.add_argument('--analysis', '-a',
						dest = 'analysis', action = 'store', nargs='+', default = None,
						metavar = str([f[0] for f in list_functions(Analyses)]),
						help = "Select analysis to perform"
	)
	parser.add_argument('--analysis-args', '-A',
						dest = 'analysis_args', action = 'store', nargs = '+', default = None,
						metavar = "x=1",
						help = "Pass on kwargs to analsis"
	)

	args = parser.parse_args()
	print args


	if isinstance(args.source, list):
		sources = args.source
	else:
		sources = [args.source]

	data = {}
	for source in sources:
		data[source]=DataPackage(source)

	if args.compare:
		if args.analysis is None:
			#Assuming NxA comparison of sources and analysis (run comparison)
			run_metric_comparison(data,args)
		else:
			raise ValueError("You're trying to do something stupid: %s"%args)
	else:
		run_overlay(data,args)

def run_metric_comparison(data, args):
	import matplotlib.pyplot as plt
	plt.rc('font', **font)
	from matplotlib.pyplot import figure, show, savefig
	from matplotlib.gridspec import GridSpec
	import Metrics

	_metrics = [Metrics.Deviation_Of_Heading,
				Metrics.PerNode_Speed,
				Metrics.PerNode_Internode_Distance_Avg]

	fig = figure()
	base_ax = fig.add_axes([0,0,1,1],)
	gs = GridSpec(len(_metrics), len(data))

	axes=[[None for _ in range(len(_metrics))] for _ in range(len(data))]
	for i,(run,d) in enumerate(data.iteritems()):
		for j,_metric in enumerate(_metrics):
			metric=_metric(data=d)
			metric.update()
			ax = fig.add_subplot(gs[j,i])
			ax.plot(metric.data, alpha=0.3)
			if metric.highlight_data is not None:
				ax.plot(metric.highlight_data, color = 'k', linestyle = '--')
			ax.grid(True, alpha='0.2')
			ax.autoscale_view(scalex = False, tight = True)
			# First Dataset Behaviour
			if i == 0:
				ax.set_ylabel(metric.label)
			# First Metric Behaviour (Title)
			if j == 0:
				ax.set_title(d.title.replace("_"," "))
			# Last Meric Behaviour (Legend)
			if j == len(_metrics)-1:
				ax.get_xaxis().set_visible(True)
				ax.set_xlabel("Time")
				if i==0:
					ax.legend(d.names, "lower center", bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure, ncol=len(d.names) )
			else:
				[l.set_visible(False) for l in ax.get_xticklabels()]
			if args.xkcd:
				ax = XKCDify(ax)
			axes[i][j]=ax
	# Now go left to right to adjust the scaling to match
	for j in range(len(axes[0])):
		(m_ymax,m_ymin)=(None,None)
		for i in range(len(axes)):
			(ymin,ymax)=axes[i][j].get_ylim()
			m_ymax=max(ymax,m_ymax)
			m_ymin=min(ymin,m_ymin)
		#Do it again to apply the row_max
		for i in range(len(axes)):
			axes[i][j].set_ylim((m_ymin,m_ymax*1.1))

	if args.dims is not None:
		fig.set_size_inches((int(d) for d in args.dims))

	resize(fig,2)

	if args.output is not None:
		savefig("%s.%s"%(args.title,args.output), bbox_inches=0)
	else:
		show()



def run_overlay(data, args):
	import pylab as pl
	pl.rc('font', **font)

	analysis = getattr(Analyses, args.analysis[0])

	fig = pl.figure()
	ax = fig.gca()

	results = {}
	for source in data.keys():
		#interactive_plot(data)
		results[source] = analysis(data=data[source])
		ax.plot(results[source][1], label=data[source].title.replace("_"," "))
	ax.legend(loc="upper right",prop={'size':12})
	ax.set_title(args.title.replace("_"," "))
	ax.set_ylabel(analysis.__name__.replace("_"," "))
	ax.set_xlabel("Time")

	if args.dims is not None:
		fig.set_size_inches((int(d) for d in args.dims))

	if args.output is not None:
		pl.savefig("%s.%s"%(args.title,args.output), bbox_inches=0)
	else:
		pl.show()


def resize(figure, scale):
	figure.set_size_inches(figure.get_size_inches()*scale)
