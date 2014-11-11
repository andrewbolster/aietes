#!/usr/bin/env python
__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP
from bounos import Analyses, _metrics

from contextlib import contextmanager
import numpy as np
import sys, os
from subprocess import call

@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout

def setup():
    exp = EXP(
              title="ThroughputTestingScenario",
              parallel=True,
              future=True,
              base_config_file='bella_static.conf'
             )

    # Scenario 1: All static
    #exp.addDefaultScenario(title="Scenario1")
    exp.addApplicationVariableScenario('app_rate', np.linspace(0.005, 0.1, 20))
    return exp


def run(exp):
    exp.run(title="ThroughputTestingScenario",
            runcount=8,
            dataFile=True)
    return exp


def set_run():
    return run(setup())

if __name__ == "__main__":
    exp = setup()
    exp = run(exp)
    logpath = "{path}/{title}.log".format(path=exp.exp_path,title=exp.title.replace(' ','_'))
    exp.dump_self()

    print("Saved detection stats to {}".format(exp.exp_path))
    os.chdir(exp.exp_path)


