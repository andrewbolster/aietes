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

def exec_comms_range(base_scenario):
    exp = EXP(
              title=base_scenario.split('.')[0],
              parallel=True,
              base_config_file=base_scenario
             )

    # Scenario 1: All static
    #exp.addDefaultScenario(title="Scenario1")
    exp.addApplicationVariableScenario('app_rate', np.linspace(0.02, 0.04, 9))


    exp.run(title="ThroughputTestingScenario",
            runcount=4,
            retain_data=False,
    )
    return exp


if __name__ == "__main__":

    base_scenarios = [
        'bella_static.conf',
        'bella_single_mobile.conf',
        'bella_allbut1_mobile.conf',
        'bella_all_mobile.conf'
        ]
    for base_scenario in base_scenarios:
        exp = exec_comms_range(base_scenario)
        logpath = "{path}/{title}.log".format(path=exp.exp_path,title=exp.title.replace(' ','_'))
        path = exp.exp_path


        print("Saved detection stats to {}".format(exp.exp_path))
        dump_trust_logs_and_stats_from_exp_paths([path], title=base_scenario.split('.')[0])


