__author__ = 'bolster'

import pandas as pd
import numpy as np
import random

from aietes.Applications import BehaviourTrust
from bounos.Analyses.Trust import generate_single_observer_trust_perspective, generate_node_trust_perspective
_ = np.seterr(invalid='ignore') # Pandas PITA Nan printing
import os
from bounos import DataPackage

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-03-16-45-26.h5"

store = pd.get_store(results_path)
trust = store.trust.xs(1, level='run', drop_level=False)
metric_list = trust.keys()
shuffled_list = list(metric_list)
random.shuffle(shuffled_list)
metric_weights=pd.Series([0,0,0,0,0,1,0,0,0], index=metric_list, dtype=float)
fake_vals = pd.Series(range(9), index=shuffled_list, dtype=float)

intermed = trust.reset_index(['target','observer'])
intermed = intermed[intermed.observer != intermed.target].reset_index().set_index(trust.index.names)

generate_node_trust_perspective(intermed, metric_weights=metric_weights, par=False)
