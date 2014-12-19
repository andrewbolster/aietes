#! /usr/bin/env python
# coding=utf-8

import os
import logging
import argparse
import warnings
from collections import OrderedDict
import re

from natsort import natsorted
import pandas as pd

from bounos import npz_in_dir, load_sources, generate_sources
from bounos.Analyses import Trust
from aietes.Tools import memory, swapsize


FORMAT = "%(asctime)-10s %(message)s"
logging.basicConfig(format=FORMAT,
                    level=logging.INFO,
                    datefmt='%H:%M:%S',
                    filename="/dev/shm/multi_loader.log")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
log = logging.getLogger(__name__)


def map_paths(paths):
    subdirs = reduce(list.__add__, [filter(os.path.isdir,
                                           map(lambda p: os.path.join(path, p),
                                               os.listdir(path)
                                           )
    ) for path in paths])
    return subdirs


def scenarios_comms(paths, generator=True):
    subdirs = natsorted(map_paths(paths))
    for i, subdir in enumerate(natsorted(subdirs)):
        title = os.path.basename(subdir)
        sources = npz_in_dir(subdir)
        subtitle = re.split('\(|\)', title)[1]
        log.info("{:.2%}:{}:{}:{:8.2f}/{:8.2f}MiB".format(
            float(i) / float(len(subdirs)),
            title, subtitle,
            memory(), swapsize()))
        if generator:
            yield (subtitle, generate_sources(sources, comms_only=True))
        else:
            yield (subtitle, load_sources(sources, comms_only=True))


def hdfstore(filename, obj):
    log.info("Storing into {}.h5".format(filename))
    store = pd.HDFStore("{}.h5".format(filename), mode='w')
    store.append(filename, object)


def generate_inverted_logs_from_paths(paths):
    logs = {}
    # Transpose per-var-per-run statistics into Per 'log' stats (i.e. rx, tx, trust, stats, etc)
    for var, runs in scenarios_comms(paths):
        for run, data in runs:

            nodes = dict(data['logs'].items() + [('stats', data['stats']), ('positions', data['positions'])])
            data = None
            run = run.split('-')[-1]
            log.info("------:{}:{:8.2f}/{:8.2f} MiB".format(
                run,
                memory(), swapsize()))
            for node, inner_logs in nodes.iteritems():
                if node in ['stats', 'positions']:
                    if not logs.has_key(node):
                        logs[node] = {}
                    if not logs[node].has_key(var):
                        logs[node][var] = {}
                    logs[node][var][run] = inner_logs
                else:
                    for k, v in inner_logs.iteritems():
                        if not logs.has_key(k):
                            logs[k] = {}
                        if not logs[k].has_key(var):
                            logs[k][var] = {}
                        if not logs[k][var].has_key(run):
                            logs[k][var][run] = {}
                        logs[k][var][run][node] = v

    return logs


def generate_dataframes_from_inverted_log(tup):
    k, v = tup
    log.info("Generating {} structure:{:8.2f}/{:8.2f} MiB".format(k, memory(), swapsize()))
    try:
        v = OrderedDict(sorted(v.iteritems(), key=lambda _k: _k))
        if k not in ['stats', 'positions']:
            # Var/Run/Node/(N/t) MultiIndex
            df = pd.concat([
                               pd.concat([
                                             pd.concat([pd.DataFrame(iiiv)
                                                        for iiik, iiiv in iiv.iteritems()],
                                                       keys=iiv.keys())
                                             for iik, iiv in iv.iteritems()],
                                         keys=iv.keys()
                               )
                               for ik, iv in v.iteritems()],
                           keys=v.keys(),
                           names=['var', 'run', 'node', 't']
            )
        elif k == 'positions':
            # Var/Run/T/Node MultiIndex
            df = pd.concat([
                               pd.concat([iiv
                                          for iik, iiv in iv.iteritems()],
                                         keys=iv.keys())
                               for ik, iv in v.iteritems()],
                           keys=v.keys(),
                           names=['var', 'run', 't', 'node']
            )
        else:
            # Var/Run MultiIndex
            df = pd.concat([
                               pd.concat([iiv
                                          for iik, iiv in iv.iteritems()],
                                         keys=iv.keys())
                               for ik, iv in v.iteritems()],
                           keys=v.keys(),
                           names=['var', 'run', 'node']
            )

        # Fixes for storage and sanity
        if k == 'stats':
            df.drop(['total_counts', u'sent_counts', 'received_counts'], axis=1, inplace=True)
        if k == 'trust':
            df = Trust.explode_metrics_from_trust_log(df)

    except:
        log.exception("{k} didn't work".format(k=k))

    return k, df


def dump_trust_logs_and_stats_from_exp_paths(paths, title=None):
    if title is None:
        title = 'logs'
    inverted_logs = generate_inverted_logs_from_paths(paths)
    log.info("First Cycle:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))
    dfs = {k: v for k, v in map(generate_dataframes_from_inverted_log, inverted_logs.iteritems())}

    filename = '{}.h5'.format(title)
    log.info("Dumping to {}:{:8.2f}/{:8.2f} MiB".format(filename, memory(), swapsize()))
    if os.path.isfile(filename):
        os.remove(filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, v in dfs.iteritems():
            v.to_hdf('{}.h5'.format(title), k, complevel=5, complib='zlib')
    log.info("Done!:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load multiple scenarios or experiments into a single output hdfstore")
    parser.add_argument('paths', metavar='P', type=str, nargs='+', default=[os.curdir], help="List of directories to walk for dataruns")
    args = parser.parse_args()
    dump_trust_logs_and_stats_from_exp_paths(args.paths)
