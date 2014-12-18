#! /usr/bin/env python
# coding=utf-8

#paths =["/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-11-12-50-42"]
paths =["/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-11-17-14-31",
        "/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-12-17-50-11"]

import polybos
from bounos import DataPackage, npz_in_dir, load_sources, generate_sources
import os, gc, logging
from natsort import natsorted
from copy import deepcopy
from bounos.Analyses.Trust import generate_trust_logs_from_comms_logs

from aietes.Tools import unpickle, is_valid_aietes_datafile, mkpickle, mkCpickle, memory, swapsize
from collections import OrderedDict
import pandas as pd
import re
import types
#exp = polybos.RecoveredExperiment.walk_dir(path)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT,level=logging.INFO,datefmt='%H:%M:%S')

log = logging.getLogger()
log.setLevel(logging.DEBUG)

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
        log.info("{:%}:{}:{}/{}".format(float(i)/float(len(subdirs)), title, memory(), swapsize()))
        yield (subdir,generate_sources(sources,comms_only=True))

def hdfstore(filename, obj):
    log.info("Storing into {}.h5".format(filename))
    store = pd.HDFStore("{}.h5".format(filename), mode='w')
    store.append(filename,object)

import ZODB, ZODB.FileStorage
import persistent, transaction
import BTrees.OOBTree

class PersistentDataPackage(DataPackage,persistent.Persistent):
    pass

storage = ZODB.FileStorage.FileStorage('mydata.fs')
db = ZODB.DB(storage)
connection = db.open()
root = connection.root

def dump_trust_logs_and_stats_from_exp_paths(paths):
    variables = root.variables = []
    trust_logs = root.trust_logs = {}
    comms_logs = root.comms_logs = {}
    for subdir, runs in scenarios_comms(paths):
        variable = re.split('\(|\)',subdir)[1]
        if not isinstance(runs, types.GeneratorType):
            runs = runs.iteritems()

        runlist = map(lambda (run, data):
                      (run.split('-')[1],deepcopy(data['stats']), deepcopy(data['logs'])),
                      runs)
        if runlist:
            variables.append(
                (variable,
                 pd.concat([r[1] for r in runlist],
                           keys=[r[0] for r in runlist],
                           names=['run','node'])
                )
            )
            trust_logs[variable]= {r[0]:generate_trust_logs_from_comms_logs(r[2]) for r in runlist}
            comms_logs[variable]= {r[0]:r[2] for r in runlist}
        transaction.commit()
        del runlist
        gc.collect()
        log.info("VAR:{}:{}/{} MiB".format(variable, memory(), swapsize()))

    gc.collect()

    log.info("Completed Collection, everything necessary should be in mydata.fs")

    try:
        df = pd.concat([v[1] for v in variables], keys=[v[0] for v in variables], names = ['variable'])
        df.drop(['received_counts','sent_counts','total_counts'], axis=1, inplace=True)

    except:
        log.exception("Dataframe Generation failed")
        raise
    log.info("Done")

