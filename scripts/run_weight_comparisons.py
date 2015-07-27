from __future__ import absolute_import

__author__ = 'bolster'

import os

import pandas as pd
import itertools

from bounos.Analyses.Weight import perform_weight_factor_analysis_on_trust_frame

phys_metrics = [u'INDD', u'INHD', u'Speed']
comms_metrics = [u'ADelay', u'ARXP', u'ATXP', u'RXThroughput', u'PLR', u'TXThroughput']


results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-03-16-45-26" # Bad Simulation Config (Wrong Default leading Random Walk to actually be comms-malicious)
results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"

if __name__ == "__main__":
    with pd.get_store(results_path + '.h5') as store:
        outliers = perform_weight_factor_analysis_on_trust_frame(store.trust, "CombinedTrust",
                                                                 min_emphasis=0, max_emphasis=3, par=True)
    outliers.to_hdf(os.path.join(results_path, "outliers.h5"), "CombinedTrust_{}_3".format("BigOne"))
