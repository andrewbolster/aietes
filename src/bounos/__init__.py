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
    for source in sources:
        data[source]=DataPackage(source)
        #interactive_plot(data)
        analyses[source] = analysis(data=data[source])
        pl.plot(analyses[source][1])
    pl.show()

