# BOUNOS - Heir to the Kingdom of AIETES

import argparse

import numpy as np
from Plotting import *
import pylab as pl
import re

np.seterr(under = "ignore")

from Metrics import *
from Analyses import *
from DataPackage import DataPackage

from aietes.Tools import list_functions
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

pl.rc('font', **font)

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
                        metavar = '2.3 2.5',
                        help = 'Figure Dimensions in Inches (default autofit)'
    )
    parser.add_argument('--analysis', '-a',dest = 'analysis', action = 'store',metavar = str([f[0] for f in list_functions(Analyses)]),help = "Select analysis to perform", required = True
    )

    args = parser.parse_args()
    print args
    if isinstance(args.source, list):
        sources = args.source
    else:
        sources = [args.source]

    analysis = getattr(Analyses, args.analysis)

    data = {}
    analyses = {}

    fig = pl.figure()
    ax = fig.gca()
    for source in sources:
        data[source]=DataPackage(source)
        #interactive_plot(data)
        analyses[source] = analysis(data=data[source])
        ax.plot(analyses[source][1], label=data[source].title.replace("_"," "))
    pl.legend(loc="upper right",prop={'size':12})
    pl.title(args.title.replace("_"," "))
    ax.set_ylabel(args.analysis.replace("_"," "))
    ax.set_xlabel("Time")

    if args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    if args.output is not None:
        pl.savefig("%s.%s"%(args.title,args.output), bbox_inches=0)
    else:
        pl.show()

