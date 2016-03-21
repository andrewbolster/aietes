#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'

from contextlib import contextmanager
import sys
import logging as log
from polybos import ExperimentManager as ExpMan
from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths


@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout


def setup_exp():
    e = ExpMan(node_count=6,
               title="Malicious Behaviour Trust Comparison",
               parallel=True,
               base_config_file="behave.conf",
               # log_level=log.INFO
               )
    e.add_minority_n_behaviour_suite(["Waypoint", "Shadow", "SlowCoach"], n_minority=1)
    return e


def run(e):
    e.run(title="4-bev-mal",
          runcount=4,
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
    try:
        exp.dump_analysis()
        with redirected(stdout=logpath):
            ExpMan.print_stats(exp, verbose=True)
        with open(logpath, 'r') as fin:
            print fin.read()
        print("Saved detection stats to {0}".format(logpath))
    except MemoryError:
        log.exception("MemErrd in dump_analysis, moving on")


    path = exp.exp_path
    print("Saved detection stats to {0}".format(exp.exp_path))
    try:
        dump_trust_logs_and_stats_from_exp_paths([path], title=exp.title)
    except:
        log.exception("Crashed in trust logging, moving on")
