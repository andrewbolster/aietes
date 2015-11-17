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


def exec_scaled_behaviour_range(base_scenarios, title, app_rate=0.025, scale=1, malice=False):
    e = ExpMan(title="{}{}-{}-{}".format(
        "Malicious{}".format(malice) if malice else "",
        title, app_rate, scale),
        parallel=False
    )
    for base_scenario in base_scenarios:
        e.add_position_scaling_range([scale], title="{}({})".format(e.title, re.split('\.|/', base_scenario)[-2]),
                                     base_scenario=base_scenario, basis_node_name="n1")
    e.update_all_nodes({"app_rate": app_rate})
    if malice:
        e.update_explicit_node("n1", {"app": malice})

    return e


if __name__ == "__main__":

    base_scenarios = [
        'bella_static.conf',
        'bella_all_mobile.conf',
        'bella_static_median.conf',
        'bella_all_mobile_median.conf'
    ]
    app_rate = 0.025
    scale = 3
    title = "TrustMedianTests"
    log = logging.getLogger()

    for malice in ["SelfishTargetSelection", "BadMouthingPowerControl", False]:
        try:
            exp = exec_scaled_behaviour_range(base_scenarios, title,
                                              app_rate=app_rate,
                                              scale=scale, malice=malice)

            exp.run(
                runcount=1,
                retain_data=False,
                queue=True,
            )

            path = exp.exp_path
            print("Saved detection stats to {}".format(exp.exp_path))
            try:
                dump_trust_logs_and_stats_from_exp_paths([path], title=exp.title)
            except:
                log.exception("Crashed in trust logging, moving on")

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            log.exception("Crashed in simulation, bailing")
