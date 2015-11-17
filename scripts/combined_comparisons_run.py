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
import run_weight_comparisons as rwc


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
          runcount=3,
          runtime=36000,
          retain_data=True,
          queue=True)
    return e


def set_run():
    return run(setup_exp())


if __name__ == "__main__":
    exp = setup_exp()
    exp = run(exp)
    logpath = "{path}/{title}.log".format(path=exp.exp_path, title=exp.title.replace(' ', '_'))
    exp.dump_analysis()

    with redirected(stdout=logpath):
        ExpMan.print_stats(exp, verbose=True)

    with open(logpath, 'r') as fin:
        print fin.read()

    print("Saved detection stats to {}".format(logpath))
    path = exp.exp_path
    print("Saved detection stats to {}".format(exp.exp_path))
    try:
        dump_trust_logs_and_stats_from_exp_paths([path], title=exp.title)
    except Exception as e:
        print("Crashed in trust logging, moving on: {}".format(traceback.format_exc()))

        # for run in range(4):
        #     with pd.get_store(exp.exp_path + '.h5') as store:
        #         sub_frame = pd.concat([
        #             store.trust.xs('Alfa', level='observer', drop_level=False),
        #             store.trust.xs('Bravo', level='observer', drop_level=False),
        #             store.trust.xs('Charlie', level='observer', drop_level=False)
        #         ]).xs(run, level='run', drop_level=False)
        #
        #     outliers = rwc.perform_weight_factor_outlier_analysis_on_trust_frame(sub_frame, "CombinedTrust", extra=run, min_emphasis=0,
        #                                                       max_emphasis=3, par=False)
        #     outliers.to_hdf(os.path.join(exp.exp_path,"outliers.h5"),"CombinedTrust_{}_3".format(run))
