# coding=utf-8
__author__ = 'bolster'
import os
import warnings
import pandas as pd
import itertools
import logging
from joblib import Parallel, delayed
from bounos.ChartBuilders import weight_comparisons
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

files = {
    "malicious": "MaliciousBadMouthingPowerControlTrustMedianTests-0.025-3-2015-02-19-23-27-01.h5",
    "good": "TrustMedianTests-0.025-3-2015-02-19-23-29-39.h5",
    "selfish": "MaliciousSelfishTargetSelectionTrustMedianTests-0.025-3-2015-03-29-19-32-36.h5",

}

trusts = {k: Trust.trust_from_file(f) for k, f in files.iteritems()}

outliers = Parallel(n_jobs=-1, verbose=10, pre_dispatch=8)(
    delayed(Trust.outliers_from_trust_dict)
    (trusts,
     par=False,
     good_key="good",
     s="bella_all_mobile",
     metric_weight=w,
     flip_metrics=['TXThroughput', 'RXThroughput', 'ATXP']
     ) for w in itertools.imap(weight_comparisons.norm_weight,
                               Trust._metric_combinations_series)
)
sums = pd.concat(outliers).reset_index()

filename = '{0}.h5'.format("/dev/shm/outliers")
log.info("Dumping to {0}:{1:8.2f}/{2:8.2f} MiB".format(filename, memory(), swapsize()))
if os.path.isfile(filename):
    os.remove(filename)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sums.to_hdf(filename, 'outliers', complevel=5, complib='zlib')
