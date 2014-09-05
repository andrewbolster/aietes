#!/usr/bin/env python
__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP
from bounos import Analyses, _metrics

from contextlib import contextmanager
import sys, os
from subprocess import call

@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout

def set():
    exp = EXP(node_count=6,
              title="Comms Behaviour Testing Scenarios",
              parallel=True,
              future=True
             )

    default_updates = {
        'behaviour':"StationKeep",
        'positioning':"randomPlane"
    }
    # Set the default behaviour to static
    exp.updateDefaultBehaviour("StationKeep")
    # Scenario 1: All static
    exp.addDefaultScenario()
    # Scenario 2: 1 mobile
    exp.addMinorityNBehaviourSuite(["RandomWalk"], n_minority=1)
    # Scenario 3: n-1 mobile
    exp.addMinorityNBehaviourSuite(["RandomWalk"], n_minority=exp.node_count-1)
    # Scenario 2: n mobile
    exp.addMinorityNBehaviourSuite(["RandomWalk"], n_minority=exp.node_count)
    return exp


def run(exp):
    exp.run(title="8-bev-mal",
            runcount=4,
            runtime=400,
            dataFile=True)
    return exp


def set_run():
    return run(set())

if __name__ == "__main__":
    exp = set()
    exp = run(exp)
    logpath = "{path}/{title}.log".format(path=exp.exp_path,title=exp.title.replace(' ','_'))
    exp.dump_self()
    exp.dump_analysis()

    with redirected(stdout=logpath):
        EXP.printStats(exp, verbose=True)

    with open(logpath, 'r') as fin:
        print fin.read()

    print("Saved detection stats to {}".format(logpath))
    os.chdir(exp.exp_path)
    call(['bounos','-M'])


