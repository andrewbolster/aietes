#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'

from contextlib import contextmanager
import sys
import os
from subprocess import call

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
               parallel=False,
               base_config_file="behave.conf"
               )
    e.add_minority_n_behaviour_suite(["Waypoint", "Shadow", "SlowCoach"], n_minority=1)
    return e


def run(e):
    e.run(title="8-bev-mal",
          runcount=8,
          runtime=1800,
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
    except:
        log.exception("Crashed in trust logging, moving on")
