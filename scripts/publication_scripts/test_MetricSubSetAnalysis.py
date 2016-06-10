# coding=utf-8

from __future__ import division

import tempfile
import logging
import unittest

import matplotlib.pylab as plt
import pandas as pd
import sklearn.ensemble as ske
from scipy.stats.stats import pearsonr

from aietes.Tools import *
from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.Analyses.Weight import target_weight_feature_extractor, \
    generate_weighted_trust_perspectives, build_outlier_weights, calc_correlations_from_weights, \
    drop_metrics_from_weights_by_key
from bounos.ChartBuilders import format_axes, latexify, unique_cm_dict_from_list

from .helpers import format_features

###########
# OPTIONS #
###########
_texcol = 0.5
_texfac = 0.9
use_temp_dir = False
show_outputs = False
recompute = True
_ = np.seterr(invalid='ignore')  # Pandas PITA Nan printing

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
w = 6

phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']
key_order = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR', 'INDD', 'INHD', 'Speed']

observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9

fig_basedir = "/home/bolster/src/thesis/Figures/"

shared_h5_path = '/dev/shm/shared_subset.h5'

"""GO TO Chapter 7 Notebook """

from aietes.Tools import uncpickle, key_order

phys_keys = ['INDD', 'INHD', 'Speed']
comm_keys = ['ADelay', 'ARXP', 'ATXP', 'RXThroughput', 'TXThroughput', 'PLR']

comm_keys_alt = ['ATXP', 'RXThroughput', 'TXThroughput', 'PLR','INDD']
phys_keys_alt = ['ADelay','ARXP', 'INDD', 'INHD', 'Speed']

def get_backup_subsets_features():
    d_subsets_feats = uncpickle(os.path.join('/home/bolster/src/aietes/scripts/notebooks/Chapters',"d_subsets_feat.pkl"))
    d_subsets_feats = dict(d_subsets_feats)
    del d_subsets_feats[('ADelay',)]
    return d_subsets_feats

def generate_subsets_features(keyorder=None, min_length=5, outlier_weights=None, outlier_path=None):
    """
    Generate the target weight features for all subsets of a given set of outlier-assessed weights
    :param keyorder: Ordered List of key strings
    :param min_length: Int, min subset length
    :param outlier_weights: DataFrame of outlier weights indexed on metrics with Behaviour-Columns; if none regenerate from outlier_path
    :param outlier_path: Experiment path to walk to build outliers
    :return:
    """
    if outlier_weights is not None and outlier_path is not None:
        raise ValueError("ONLY one of outlier_weights / outlier_path must be defined")
    elif outlier_path is not None:
        logging.debug("loading from path, this may take some time")
        outlier_weights = build_outlier_weights(outlier_path, observer=observer, target=target, n_metrics=n_metrics)
    elif outlier_weights is not None:
        logging.debug("loading from df")
    else:
        raise ValueError("One of outlier_weights / outlier_path must be defined")

    # Process identical to metric subset_weight_and_feature_extractor in Thesis_diagrams


