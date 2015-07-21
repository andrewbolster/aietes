from __future__ import absolute_import

__author__ = 'bolster'

from time import gmtime, strftime

import pandas as pd
import itertools
from joblib import Parallel, delayed

from bounos.Analyses.Weight import generate_outlier_frame
from bounos.ChartBuilders.weight_comparisons import norm_weight

phys_metrics = [u'INDD', u'INHD', u'Speed']
comms_metrics = [u'ADelay', u'ARXP', u'ATXP', u'RXThroughput', u'PLR', u'TXThroughput']


results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-03-16-45-26"

def perform_weight_factor_analysis_on_trust_frame(trust_frame, good, min_emphasis = 0, max_emphasis = 3, extra=None, verbose=False, par=False):
    """
    For a given trust frame perform grey weight factor distribution analysis to
    generate a frame with metrics/metric weights as a multiindex with columns for each comparison between behaviours

    :param trust_frame:
    :param good: the baseline behaviour to assess against
    :param min_metrics:
    :param: max_metrics:
    :param: max_emphasis:
    :return:
    """
    # Extract trust metric names from frame
    trust_metrics = list(trust_frame.keys())
    if max_emphasis < 2:
        raise RuntimeError("Setting Max Emphasis <2 is pointless; Runs all Zeros")

    if extra is None:
        extra = ""

    #run_vars = set(trust_frame.index.levels[trust_frame.index.names.index('var')])
    combinations = itertools.product(xrange(min_emphasis, max_emphasis), repeat=len(trust_metrics))
    if par:
        outliers = _outlier_par_inner_single_thread(combinations, good, trust_frame, trust_metrics, verbose=True)
    else:
        outliers = _outlier_single_thread_inner_par(combinations, good, trust_frame, trust_metrics, verbose=True)

    sums = pd.concat(outliers).reset_index()
    sums.to_hdf('/home/bolster/src/aietes/results/outlier_backup.h5', "{}{}_{}".format(good,extra,max_emphasis))
    return sums


def _outlier_single_thread_inner_par(combinations, good, trust_frame, trust_metrics, verbose):
    outliers = []
    for i, w in enumerate(combinations):
        if verbose: print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), i, w)
        outlier = generate_outlier_frame(good, trust_frame, norm_weight(w, trust_metrics), par=True)
        outliers.append(outlier)
    return outliers

def _outlier_par_inner_single_thread(combinations, good, trust_frame, trust_metrics, verbose):

    outliers = Parallel(n_jobs=-1, verbose=int(verbose))(delayed(generate_outlier_frame)
                                   (good, trust_frame, norm_weight(w, trust_metrics), False)
                                   for w in combinations)
    return outliers

if __name__ == "__main__":
    run = 1
    with pd.get_store(results_path + '.h5') as store:
        sub_frame = pd.concat([
            store.trust.xs('Alfa', level='observer', drop_level=False),
            store.trust.xs('Bravo', level='observer', drop_level=False),
            store.trust.xs('Charlie', level='observer', drop_level=False)
        ]).xs(run, level='run', drop_level=False)

    perform_weight_factor_analysis_on_trust_frame(sub_frame, "RandomFlatWalk", extra=run, min_emphasis=0, max_emphasis=3, par=False)
