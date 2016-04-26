# coding=utf-8
import unittest
import matplotlib.pylab as plt
from shutil import copy
import pandas as pd
from tqdm import *
from joblib import Parallel, delayed
from helpers import best_of_all

import sklearn.ensemble as ske
from scipy.stats.stats import pearsonr

from aietes.Tools import *

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
json_path = "/home/bolster/Dropbox/aietes_json_results"
fig_basedir = "/home/bolster/src/thesis/Figures/"

shared_h5_path = '/dev/shm/shared.h5'

def run_all_analysis_generation(results_path):
    with pd.get_store(results_path + '.h5') as store:
        trust_observations = store.trust.dropna()


    def power_set_map(powerset_string):
        strings = powerset_string[1:].split('_')
        metrics = strings[:-3]
        t = strings[-2]
        signed = strings[-1]
        return ({
            'metrics': metrics,
            'type': t,
            'dataset': powerset_string[1:],
            'signed': signed
        })

    shared_big_h5_path = "/home/bolster/src/aietes/results/powerset_shared.h5"
    with pd.get_store(shared_big_h5_path) as store:
        shuffledkeys = random.sample(store.keys(), len(store.keys()))
        result = map(power_set_map, shuffledkeys)
        powerset_list = filter(lambda t: t['type'] == 'feats', result)
        random.shuffle(powerset_list)
        for d in powerset_list:
            d['data'] = store[d['dataset']]

    best_weight_valences_and_runs_for_metric_subset = {}
    with Parallel(n_jobs=-1) as par:
        for subset_d in tqdm(powerset_list): # This maps to a single powerset_shared.h5 dataset (comms_alt, etc)
            best_output_file = '{}.json'.format(os.path.join(json_path,subset_d['dataset']))

            if os.path.isfile(best_output_file):
                print("Skipping: File already exists, consider deleting it {}".format(best_output_file))
                continue
            try:
                feat_weights = categorise_dataframe(non_zero_rows(subset_d['data']).T)
                if subset_d['metrics'] is not None:
                    best = best_of_all(feat_weights, trust_observations[subset_d['metrics']], par=par)
                else:
                    best = best_of_all(feat_weights, trust_observations, par=par)
                best_weight_valences_and_runs_for_metric_subset[subset_d['dataset']] = best


                with open(best_output_file, 'w') as f:
                    json.dump(best, f, cls=NumpyAwareJSONEncoder)
                copy(best_output_file, results_path)
            except:
                print("Failed on {}".format(subset_d['dataset']))
                raise

    return best_weight_valences_and_runs_for_metric_subset

if __name__ == "__main__":
    result = run_all_analysis_generation(results_path)

    mkcpickle("/home/bolster/src/aietes/results/best_weight_valences_and_runs_for_metric_powerset",
              result)

