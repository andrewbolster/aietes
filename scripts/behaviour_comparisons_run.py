#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'

from contextlib import contextmanager
import sys
import os
from subprocess import call

from polybos import ExperimentManager as EXP


@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout


def setup_exp():
    e = EXP(node_count=8,
            title="Malicious Behaviour Trust Comparison",
            parallel=True,
            future=True
    )
    e.add_minority_n_behaviour_suite(["Waypoint", "Shadow", "SlowCoach"], n_minority=1)
    return e


def run(e):
    e.run(title="8-bev-mal",
          runcount=4,
          runtime=400,
          dataFile=True)
    return e


def set_run():
    return run(setup_exp())


if __name__ == "__main__":
    exp = setup_exp()
    exp = run(exp)
    logpath = "{path}/{title}.log".format(path=exp.exp_path, title=exp.title.replace(' ', '_'))
    exp.dump_analysis()

    with redirected(stdout=logpath):
        EXP.print_stats(exp, verbose=True)

    with open(logpath, 'r') as fin:
        print fin.read()

    print("Saved detection stats to {}".format(logpath))
    exp.dump_self()
    os.chdir(exp.exp_path)
    call(['bounos', '-M'])


