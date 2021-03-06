#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'

from contextlib import contextmanager
import sys
import re
import logging
import numpy as np
from polybos import ExperimentManager as ExpMan
from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths

logging.basicConfig()

@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout


def exec_comms_range(scenario, title, app_rate):
    e = ExpMan(title="{0}-{1}-{2}".format(title, re.split('\.|/', scenario)[-2], app_rate),
               parallel=True,
               base_config_file=scenario,
               log_level=logging.WARN
               )
    e.add_position_scaling_range(np.linspace(1, 16, 32), basis_node_name="n1")
    e.update_all_nodes({"app_rate": app_rate})
    e.run(
        runcount=8,
        retain_data=False,
        queue=True
    )
    return e


if __name__ == "__main__":

    base_scenarios = [
        'bella_static.conf',
        'bella_single_mobile.conf',
        'bella_allbut1_mobile.conf',
        'bella_all_mobile.conf'
    ]
    app_range = np.arange(0.005, 0.035, step=0.001).tolist()
    title = "CommsRateAndRangeTest"
    log = logging.getLogger()
    if len(sys.argv) > 1:
        N = int(sys.argv[-2])
        n = int(sys.argv[-1])
        log.info("Generating section {0} of {1} for {2}".format(n, N, title))
        span = len(app_range) / N
        init = n * span
        app_range = app_range[init:init + span]
    for base_scenario in base_scenarios:
        for app_rate in list(sorted(app_range)):
            try:
                exp = exec_comms_range(base_scenario, title, app_rate)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                log.exception("Crashed in simulation, moving on")
                continue
            path = exp.exp_path
            print("Saved detection stats to {0}".format(exp.exp_path))
            base_name = re.split('\.|/', base_scenario)[-2]
            try:
                dump_trust_logs_and_stats_from_exp_paths([path],
                                                         title="{0}-{1}-{2:.4f}".format(title, base_name, app_rate))
            except:
                log.exception("Crashed in trust logging, moving on")
