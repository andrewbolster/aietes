#!/usr/bin/env python
__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP

from contextlib import contextmanager
import numpy as np
import sys, os
import re

from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths

import logging
logging.basicConfig()

@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout

def exec_comms_range(base_scenario, title):
    exp = EXP(title="{}-{}".format(title,re.split('\.|\/',base_scenario)[-2]),
              parallel=True,
              base_config_file=base_scenario
             )

    value_range = set(np.arange(0.02,0.035,step=0.0025).tolist() + np.arange(0.015,0.075, step=0.005).tolist())
    exp.addApplicationVariableScenario('app_rate',value_range)

    exp.run(
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
    title="CommsRateTest"
    log=logging.getLogger()
    for base_scenario in base_scenarios:
        try:
            exp = exec_comms_range(base_scenario, title)
        except:
            log.exception("Crashed in simulation, moving on")
            continue
        path = exp.exp_path
        print("Saved detection stats to {}".format(exp.exp_path))
        base_name = re.split('\.|\/',base_scenario)[-2]
        try:
            dump_trust_logs_and_stats_from_exp_paths([path], title="{}-{}".format(title,base_name))
        except:
            log.exception("Crashed in trust logging, moving on")

