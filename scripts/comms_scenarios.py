#!/usr/bin/env python
__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP
from bounos import Analyses, _metrics

from contextlib import contextmanager
import numpy as np
import sys, os

from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths


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
              base_config_file='bella_static.conf'
             )

    # Scenario 1: All static
    #exp.addDefaultScenario(title="Scenario1")
    exp.addApplicationVariableScenario('app_rate', np.linspace(0.01, 0.06, 20))
    return exp


def run(exp):
    exp.run(title="ThroughputTestingScenario",
            runcount=4,
            retain_data=False,
    )
    return exp



if __name__ == "__main__":
    exp = set(run())
    logpath = "{path}/{title}.log".format(path=exp.exp_path,title=exp.title.replace(' ','_'))
    exp.dump_self()
    path = exp.exp_path


    print("Saved detection stats to {}".format(exp.exp_path))
    dump_trust_logs_and_stats_from_exp_paths([path])


