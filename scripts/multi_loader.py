#! /usr/bin/env python

paths =["/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-11-17-14-31",
        "/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-12-17-50-11"]
paths =["/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-11-12-50-42"]

import polybos
reload(polybos)
from bounos import DataPackage, npz_in_dir, load_sources
import os, gc
from natsort import natsorted
from copy import deepcopy

from aietes.Tools import unpickle, is_valid_aietes_datafile, mkpickle
import pandas as pd
import re
#exp = polybos.RecoveredExperiment.walk_dir(path)

def grab_comms(s):
    dp = DataPackage(s)
    return dp.comms
def map_paths(paths):
    subdirs=reduce(list.__add__,[filter(os.path.isdir,
               map(lambda p: os.path.join(path,p),
                   os.listdir(path)
               )
        ) for path in paths])
    return subdirs

def scenarios_comms(paths):
    subdirs = natsorted(map_paths(paths))
    for i,subdir in enumerate(natsorted(subdirs)):
        title = os.path.basename(subdir)
        sources = npz_in_dir(subdir)
        print("{:%}:{}".format(float(i)/float(len(subdirs)), subdir))
        yield (subdir,load_sources(sources,comms_only=True))

def hdfstore(filename, obj):
    print("Storing into {}.h5".format(filename))
    store = pd.HDFStore("{}.h5".format(filename), mode='w')
    store.append(filename,object)

if __name__ == "__main__":

    variables = []
    logs = []
    for subdir, runs in scenarios_comms(paths):
        variable = re.split('\)|\)',subdir)[1]
        runlist = map(lambda (run, data): (run.split('-')[1],deepcopy(data['stats']), deepcopy(data['logs'])),
                      runs.iteritems())
        if runlist:
            variables.append(
                (variable,
                 pd.concat([r[1] for r in runlist],
                           keys=[r[0] for r in runlist],
                           names=['run','node'])
                )
            )
            logs.append(
                (variable,
                 {r[0]:r[2] for r in runlist},
                )
            )
        gc.collect()

    mkpickle("logs.pkl",logs)
    try:
        df = pd.concat([v[1] for v in variables], keys=[v[0] for v in variables], names = ['variable'])
        df.drop(['received_counts','sent_counts','total_counts'], axis=1, inplace=True)
    except:
        mkpickle("stats.pkl",variables)
        raise

    hdfstore('df',df)
