#!/usr/bin/env python
# coding=utf-8
import traceback

__author__ = 'andrewbolster'

from contextlib import contextmanager
import sys
import pandas as pd
import os
from subprocess import call
from polybos import ExperimentManager as ExpMan
from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths
import bounos.Analyses.Weight as Weight


@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout


def setup_exp():
    e = ExpMan(node_count=6,
               title="Malicious Behaviour Trust Control",
               parallel=True,
               base_config_file="combined.conf"
               )
    # Combined Trust acts as baseline for both comms and behaviour level
    e.add_minority_n_application_suite(
        ["CombinedBadMouthingPowerControl", "CombinedSelfishTargetSelection", "CombinedTrust", ], n_minority=1)
    e.add_minority_n_behaviour_suite(["Shadow", "SlowCoach"], n_minority=1)

    return e


def run(e):
    e.run(title="8-bev-mal",
          runcount=4,
          runtime=18000,
          retain_data=True,
          queue=True)
    return e


def set_run():
    return run(setup_exp())


def generate_mean_delta_t(path):
    with pd.get_store(path + '.h5') as store:
        trust_frame = store.trust

    outliers = Weight.perform_weight_factor_target_mean_t_delta_analysis_on_trust_frame(trust_frame, min_emphasis=0,
                                                                                        max_emphasis=2, max_sum=1,
                                                                                        verbose=False, par=True)
    outliers.to_hdf(os.path.join(path, "outliers.h5"),
                "{0}".format("meandelta"))
    return outliers

if __name__ == "__main__":
    exp = setup_exp()
    exp = run(exp)
    logpath = "{path}/{title}.log".format(path=exp.exp_path, title=exp.title.replace(' ', '_'))
    #exp.dump_analysis()

    print("Saved detection stats to {0}".format(logpath))
    path = exp.exp_path
    print("Saved detection stats to {0}".format(exp.exp_path))
    try:
        dump_trust_logs_and_stats_from_exp_paths([path], title=exp.title)
        generate_mean_delta_t(path)
    except Exception as e:
        print("Crashed in trust logging, moving on: {0}".format(traceback.format_exc()))

