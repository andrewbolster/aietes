# coding=utf-8
from __future__ import absolute_import

__author__ = 'bolster'

import os
import pandas as pd
import itertools
from bounos.Analyses.Weight import perform_weight_factor_outlier_analysis_on_trust_frame

phys_metrics = [u'INDD', u'INHD', u'Speed']
comms_metrics = [u'ADelay', u'ARXP', u'ATXP', u'RXThroughput', u'PLR', u'TXThroughput']

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-03-16-45-26"  # Bad Simulation Config (Wrong Default leading Random Walk to actually be comms-malicious)
if os.path.isdir('/mnt/fast/aietes_results'):
    results_path = "/mnt/fast/aietes_results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
else:
    results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"

if __name__ == "__main__":
    with pd.get_store(results_path + '.h5') as store:
        trust = store.trust.xs('Bravo', level='observer', drop_level=False).dropna()

    outliers = perform_weight_factor_outlier_analysis_on_trust_frame(trust, "CombinedTrust",
                                                                     min_emphasis=-1, max_emphasis=2, par=True)
    outliers.to_hdf(os.path.join(results_path, "outliers.h5"), "CombinedTrust_{0}_3".format("Signed"))
