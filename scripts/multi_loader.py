#! /usr/bin/env python

#paths = ["/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-11-12-50-42"]
paths =["/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-11-17-14-31",
       "/home/bolster/src/aietes/results/ThroughputTestingScenario-2014-11-12-17-50-11"]

import os
import logging
import warnings
from collections import OrderedDict
import re

from natsort import natsorted
import pandas as pd

from bounos import npz_in_dir, load_sources, generate_sources
from aietes.Tools import memory, swapsize

FORMAT = "%(asctime)-10s %(message)s"
logging.basicConfig(format=FORMAT,
                    level=logging.INFO,
                    datefmt='%H:%M:%S',
                    filename="/dev/shm/multi_loader.log")

log = logging.getLogger()

def map_paths(paths):
    subdirs = reduce(list.__add__, [filter(os.path.isdir,
                                           map(lambda p: os.path.join(path, p),
                                               os.listdir(path)
                                           )
    ) for path in paths])
    return subdirs

def scenarios_comms(paths, generator = True):
    subdirs = natsorted(map_paths(paths))
    for i,subdir in enumerate(natsorted(subdirs)):
        title = os.path.basename(subdir)
        sources = npz_in_dir(subdir)
        subtitle=re.split('\(|\)',title)[1]
        log.info("{:.2%}:{}:{}:{:8.2f}/{:8.2f}MiB".format(
            float(i)/float(len(subdirs)),
            title, subtitle,
            memory(), swapsize()))
        if generator:
            yield (subtitle,generate_sources(sources,comms_only=True))
        else:
            yield (subtitle,load_sources(sources,comms_only=True))


def hdfstore(filename, obj):
    log.info("Storing into {}.h5".format(filename))
    store = pd.HDFStore("{}.h5".format(filename), mode='w')
    store.append(filename, object)


def dump_trust_logs_and_stats_from_exp_paths(paths):
    inverted_logs = {}
    # Transpose per-var-per-run statistics into Per 'log' stats (i.e. rx, tx, trust, stats, etc)
    for var, runs in scenarios_comms(paths):
        for run, data in runs:
            nodes = dict(data['logs'].items() + [('stats', data['stats'])])
            data = None
            run = run.split('-')[-1]
            log.info("------:{}:{:8.2f}/{:8.2f} MiB".format(
                     run,
                     memory(), swapsize()))
            for node, inner_logs in nodes.iteritems():
                if node == 'stats':
                    if not inverted_logs.has_key(node):
                        inverted_logs[node] = {}
                    if not inverted_logs[node].has_key(var):
                        inverted_logs[node][var] = {}
                    inverted_logs[node][var][run] = inner_logs
                else:
                    for k, v in inner_logs.iteritems():
                        if not inverted_logs.has_key(k):
                            inverted_logs[k] = {}
                        if not inverted_logs[k].has_key(var):
                            inverted_logs[k][var] = {}
                        if not inverted_logs[k][var].has_key(run):
                            inverted_logs[k][var][run] = {}
                        inverted_logs[k][var][run][node] = v
    dfs = {}
    log.info("First Cycle:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))
    for k, v in inverted_logs.iteritems():
        log.info("Generating {} structure:{:8.2f}/{:8.2f} MiB".format(k,memory(), swapsize()))
        try:
            v = OrderedDict(sorted(v.iteritems(), key=lambda _k: _k))
            if k not in ['stats']:
                # Var/Run/Node/(N/t) MultiIndex
                dfs[k] = pd.concat([
                                       pd.concat([
                                                     pd.concat([pd.DataFrame(iiiv)
                                                                for iiik, iiiv in iiv.iteritems()],
                                                               keys=iiv.keys())
                                                     for iik, iiv in iv.iteritems()],
                                                 keys=iv.keys()
                                       )
                                       for ik, iv in v.iteritems()],
                                   keys=[ik for ik, iv in v.iteritems()],
                                   names=['var', 'run', 'node', 'n']
                )
            else:
                # Var/Run MultiIndex
                dfs[k] = pd.concat([
                                       pd.concat([iiv
                                                  for iik, iiv in iv.iteritems()],
                                                 keys=iv.keys())
                                       for ik, iv in v.iteritems()],
                                   keys=v.keys(),
                                   names=['var', 'run', 'node']
                )
        except:
            log.info("{k} didn't work".format(k=k))
            raise
    log.info("Dumping to logs.h5:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, v in dfs.iteritems():
            v.to_hdf('logs.h5', k)
    log.info("Done!:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))


if __name__ == "__main__":
    dump_trust_logs_and_stats_from_exp_paths(paths)
