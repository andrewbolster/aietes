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
import numpy as np

from bounos import npz_in_dir, load_sources, generate_sources
from bounos.Analyses import Trust
from aietes.Tools import memory, swapsize, map_paths, mkcpickle

FORMAT = "%(asctime)-10s %(message)s"
logging.basicConfig(format=FORMAT,
                    level=logging.INFO,
                    datefmt='%H:%M:%S',
                    filename="/dev/shm/multi_loader.log")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
log = logging.getLogger(__name__)


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
    logs = {} #TODO Update this to a defaultdict implementation.
    # Transpose per-var-per-run statistics into Per 'log' stats (i.e. rx, tx, trust, stats, etc)
    for var, runs in scenarios_comms(paths):
        print("Var:{}".format(var))
        for run, data in runs:
            print("---{}".format(run))
            nodes = dict(data['logs'].items() + [('stats', data['stats']), ('positions', data['positions'])])
            data = None
            run = run.split('-')[-1]
            log.info("------:{}:{:8.2f}/{:8.2f} MiB".format(
                run,
                memory(), swapsize()))
            for node, inner_logs in nodes.iteritems():
                if node in ['stats', 'positions']:
                    if node not in logs:
                        logs[node] = {}
                    if var not in logs[node]:
                        logs[node][var] = {}
                    logs[node][var][run] = inner_logs
                else:
                    for k, v in inner_logs.iteritems():
                        if k not in logs:
                            logs[k] = {}
                        if var not in logs[k]:
                            logs[k][var] = {}
                        if run not in logs[k][var]:
                            logs[k][var][run] = {}
                        logs[k][var][run][node] = v

    return logs


def generate_dataframes_from_inverted_log(tup):
    k, v = tup
    log.info("Generating {} structure:{:8.2f}/{:8.2f} MiB".format(k, memory(), swapsize()))
    try:
        v = OrderedDict(sorted(v.iteritems(), key=lambda _k: _k))
        if k not in ['stats', 'positions', 'trust']:
            # Var/Run/Node/(N/t) MultiIndex with forgiveness for variable-length inner arrays
            df = pd.concat([
                pd.concat([
                              pd.concat([pd.DataFrame(iiiv)
                               for iiik, iiiv in iiv.iteritems()],
                              keys=iiv.keys())
                    for iik, iiv in _runs.iteritems()],
                    keys=_runs.keys()
                )
                for _var, _runs in v.iteritems()],
                keys=v.keys(),
                names=['var', 'run', 'node', 't']
            )
        elif k in ['trust']:
            # Strict Var/Run/Node/t MultiIndex
            df = pd.concat([
                pd.concat([
                    pd.concat([pd.DataFrame(dict([ (_k,pd.Series(_v)) for _k,_v in iiiv.iteritems() ]))
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
                           for iik, iiv in _runs.iteritems()],
                          keys=_runs.keys())
                for _var, _runs in v.iteritems()],
                keys=v.keys(),
                names=['var', 'run', 't', 'node']
            )
        else:
            # Var/Run MultiIndex
            df = pd.concat([
                pd.concat([iiv
                           for iik, iiv in _runs.iteritems()],
                          keys=_runs.keys())
                for _var, _runs in v.iteritems()],
                keys=v.keys(),
                names=['var', 'run', 'node']
            )

        # Fixes for storage and sanity
        if k == 'stats':
            df.drop(['total_counts', u'sent_counts', 'received_counts'],
                    axis=1, inplace=True, errors='ignore')

        if k == 'trust':
            df = Trust.explode_metrics_from_trust_log(df)

        # Ensure Index Types and Orders
        try:
            map(float, df.index.levels[0])
            var_is_float = True
        except:
            var_is_float = False

        df.index = df.index.set_levels([
            df.index.levels[0].astype(np.float64) if var_is_float else df.index.levels[
                0],  # Var
            df.index.levels[1].astype(np.int32)  # Run
        ] + (df.index.levels[2:])
        )
        df = df.reindex(sorted(df.index.levels[0]), level=0, copy=False)  # Var
        df = df.reindex(sorted(df.index.levels[1]), level=1, copy=False)  # Var

    # TODO Give this a bloody exception clause
    except:
        log.exception("{k} didn't work".format(k=k))
        raise

    return k, df

def trust_frames_from_logs(inverted_logs):
    return {k: v for k, v in map(generate_dataframes_from_inverted_log, inverted_logs.iteritems())}


def dump_trust_logs_and_stats_from_exp_paths(paths, title=None, dump=False):
    if title is None:
        title = 'logs'

    print("Using {} as title".format(title))

    if dump:
        print("Dumping intermediates to intermediate_{}_*.npz".format(title))
    inverted_logs = generate_inverted_logs_from_paths(paths)
    if dump:
        mkcpickle("intermediate_{}_inverted_logs".format(title),inverted_logs)
    log.info("First Cycle:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))
    dfs = trust_frames_from_logs(inverted_logs)
    filename = '{}.h5'.format(title)
    log.info("Dumping to {}:{:8.2f}/{:8.2f} MiB".format(filename, memory(), swapsize()))
    if os.path.isfile(filename):
        os.remove(filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, v in dfs.iteritems():
            v.to_hdf('{}.h5'.format(title), k, complevel=5, complib='zlib')
    log.info("Done!:{:8.2f}/{:8.2f} MiB".format(memory(), swapsize()))

def results_path_parser(path):
    # Stretch and then compress the path to make sure we're in the right place
    abspath = os.path.abspath(path)
    assert os.path.isdir(path), "Expected a path! got {}".format(path)
    s=os.path.basename(abspath)
    args = s.split('-')
    title = args[0]
    scenario = args[1]
    var = args[2]
    #date = datetime(*map(int,args[3:]))
    return title, scenario, var#, date

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load multiple scenarios or experiments into a single output hdfstore")
    parser.add_argument('paths', metavar='P', type=str, nargs='+', default=[os.curdir],
                        help="List of directories to walk for dataruns")
    parser.add_argument('--title', metavar='T', type=str, default=None,
                        help="Title")
    parser.add_argument('--dump', action='store_true', default=False,
                        help="Dump intermediate files to the current directory as intermediate_*.npz")
    parser.add_argument('--infer', action='store_true', default=False,
                        help="Attempt to guess the title format from the given paths")
    args = parser.parse_args()
    for path in args.paths:
        if args.infer:
            if args.title is not None:
                raise RuntimeError("Can't infer and set the title at the same time!")
            title, base_name, var = results_path_parser(path)
        dump_trust_logs_and_stats_from_exp_paths([path], dump=args.dump,
                                                 title="{}-{}-{:.4f}".format(title, base_name, var))
