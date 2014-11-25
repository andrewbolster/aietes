__author__ = 'bolster'
import numpy as np
import matplotlib.pyplot as plt
import polybos

reload(polybos)
from bounos import DataPackage, npz_in_dir, load_sources, generate_sources
import os, gc
from natsort import natsorted
from copy import deepcopy
import bounos.Analyses.Trust as Trust
import seaborn as sns

from aietes.Tools import unpickle, is_valid_aietes_datafile, mkpickle, mkCpickle, unCpickle
from collections import OrderedDict
import pandas as pd
import re
# exp = polybos.RecoveredExperiment.walk_dir(path)

def grab_comms(s):
    dp = DataPackage(s)
    return dp.comms

def map_paths(paths):
    subdirs = reduce(list.__add__, [filter(os.path.isdir,
                                           map(lambda p: os.path.join(path, p),
                                               os.listdir(path)
                                           )
    ) for path in paths])
    return subdirs

def scenarios_comms(paths, generator=False):
    subdirs = natsorted(map_paths(paths))
    for i, subdir in enumerate(natsorted(subdirs)):
        title = os.path.basename(subdir)
        sources = npz_in_dir(subdir)
        print("{:%}:{}".format(float(i) / float(len(subdirs)), subdir))
        if generator:
            yield (subdir, generate_sources(sources, comms_only=True))
        else:
            yield (subdir, load_sources(sources, comms_only=True))

def hdfstore(filename, obj):
    print("Storing into {}.h5".format(filename))
    store = pd.HDFStore("{}.h5".format(filename), mode='w')
    try:
        store.append(filename, object, data_columns=True)
    finally:
        store.close()

logs = unCpickle('trust_logs.pkl')
logs = OrderedDict(sorted(logs.iteritems(), key=lambda k: k))

def network_trust_dict(trust_inverted):
    fs = [lambda x: -x + 1,
          lambda x: -2 * x + 2 if x > 0.5 else 2 * x,
          lambda x: x]
    sigmas = [0.0, 0.5, 1.0]
    whitenized = lambda x: map(lambda f: f(x), fs)
    white_class = lambda x: (whitenized(x).index(max(whitenized(x))))

    observer = 'n0'
    recommendation_nodes = ['n2', 'n3']
    target = 'n1'
    indirect_nodes = ['n4', 'n5']
    t_direct = lambda x: 0.5 * max(whitenized(x)) * x
    t_recommend = lambda x: 0.5 * (2 * len(recommendation_nodes) / (2.0 * len(recommendation_nodes) + len(indirect_nodes))) * max(whitenized(x)) * x
    t_indirect = lambda x: 0.5 * (2 * len(indirect_nodes) / (2.0 * len(recommendation_nodes) + len(indirect_nodes))) * max(whitenized(x)) * x

    def network_trust(t):
        t_sum = 0
        who = [observer] + recommendation_nodes + indirect_nodes
        for whom in who:
            t_sum += trust_inverted[whom][target][t]
        return t_sum / float(len(who))

    def total_trust(t):
        Td = t_direct(trust_inverted[observer][target][t])
        Tr = np.average([t_recommend(trust_inverted[recommender][target][t]) for recommender in recommendation_nodes])
        Ti = np.average([t_indirect(trust_inverted[indirecter][target][t]) for indirecter in indirect_nodes])
        return sum((Td, Tr, Ti))

    tmax = len(trust_inverted[observer][target])

    T_total = map(total_trust, range(1, tmax))
    T_network = map(network_trust, range(1, tmax))
    T_class = map(white_class, T_total)

    _d = pd.DataFrame.from_dict(
        {"t10": pd.Series(trust_inverted[observer][target]),
         "t12": pd.Series(trust_inverted['n2'][target]),
         "t13": pd.Series(trust_inverted['n3'][target]),
         "t14": pd.Series(trust_inverted['n4'][target]),
         "t15": pd.Series(trust_inverted['n5'][target]),
         "t10-5": pd.Series(T_total),
         "t10-net": pd.Series(T_network)}
    )
    return _d

def generate_trust_perspectives_from_logs(logs, metric_weights=None):
    rate_collector = []
    for rate, sim_runs in logs.iteritems():
        run_collector = []
        for i, trust_logs in sim_runs.items():
            trust_perspectives = {
                node: Trust.generate_node_trust_perspective(node_observations, metric_weights=metric_weights)
                for node, node_observations in trust_logs.items()
            }
            inverted_trust_perspectives = {
                node: Trust.invert_node_trust_perspective(node_perspective)
                for node, node_perspective in trust_perspectives.items()
            }
            run_collector.append(network_trust_dict(inverted_trust_perspectives))
        rate_collector.append((rate, pd.concat(run_collector, names=['run'])))
    return trust_perspectives, inverted_trust_perspectives, rate_collector

import itertools

trust_metrics = np.asarray("ATXP,ARXP,ADelay,ALength,Throughput,PLR".split(','))
trust_combinations = []
map(trust_combinations.extend, np.asarray([itertools.combinations(trust_metrics, i) for i in range(2, len(trust_metrics))]))
trust_combinations = np.asarray(trust_combinations)
#print trust_combinations
trust_metric_selections = np.asarray([map(lambda m: float(m in trust_combination), trust_metrics) for trust_combination in trust_combinations])
trust_metric_weights = map(lambda s: s / sum(s), trust_metric_selections)


def gen_trust_plots_for_weights(metric_weight=None):
    trust_perspectives, inverted_trust_perspectives, rate_collector = generate_trust_perspectives_from_logs(logs, metric_weights=metric_weight)

    rate_frame = pd.concat([v for _, v in rate_collector], keys=[v for v, _ in rate_collector], names=['variable', 't'])
    #sns.boxplot(rate_frame,showmeans=True, showbox=False, widths = 0.2, linewidth=2)
    #rate_frame.reset_index(level=['variable'],inplace=True)
    f, ax = plt.subplots()
    f.set_size_inches(13, 6)
    print w

    vals = rate_frame.dropna().groupby(level=['variable'], sort=True)
    #vals.boxplot(layout=(2,5), ax=ax)
    vals.get_group('0.025').boxplot(ax=ax)
    if metric_weight is not None:
        f.suptitle(",".join(trust_metrics[np.where(metric_weight > 0)].tolist()))
    else:
        f.suptitle("Utilising all Trust Metrics")
    return f


for w in trust_metric_weights[0:10:2]:
    f=gen_trust_plots_for_weights(metric_weight=w)
    plt.draw()
plt.show()

