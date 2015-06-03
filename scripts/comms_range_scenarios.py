#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'

from contextlib import contextmanager
import sys
import re
import logging

import numpy as np

from polybos import ExperimentManager as EXP
from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths

logging.basicConfig()


@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout


def exec_comms_range(scenario, title):
    e = EXP(title="{}-{}".format(title, re.split('\.|\/', scenario)[-2]),
            parallel=True,
            base_config_file=scenario
            )

    e.add_position_scaling_range(np.linspace(1, 10, 19), basis_node_name="n1")
    e.run(
        runcount=4,
        retain_data=False
    )
    return e


if __name__ == "__main__":
    base_scenarios = [
        'bella_static.conf',
        'bella_single_mobile.conf',
        'bella_allbut1_mobile.conf',
        'bella_all_mobile.conf'
    ]
    title = "CommsRangeTest"
    log = logging.getLogger()
    for base_scenario in base_scenarios:
        try:
            exp = exec_comms_range(base_scenario, title)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            log.exception("Crashed in simulation, moving on")
            continue
        path = exp.exp_path
        print("Saved detection stats to {}".format(exp.exp_path))
        base_name = re.split('\.|\/', base_scenario)[-2]
        try:
            dump_trust_logs_and_stats_from_exp_paths([path], title="{}-{}".format(title, base_name))
        except:
            log.exception("Crashed in trust logging, moving on")



