#!/usr/bin/env python
__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP
from bounos import Analyses, _metrics

from contextlib import contextmanager
import numpy as np
import sys, os
import re

from bounos.multi_loader import dump_trust_logs_and_stats_from_exp_paths


@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout

def exec_comms_range(base_scenario):
    exp = EXP(title="CommsRangeTest-{}".format(re.split('\.|\/',base_scenario)[-2]),
              parallel=True,
              base_config_file=base_scenario
    )

    exp.addPositionScalingRange(np.linspace(1,4,16), basis_node_name="n1")
    exp.run(
        runcount=1,
        retain_data=False
    )

if __name__ == "__main__":
    base_scenario = 'bella_static.conf'
    base_name = re.split('\.|\/',base_scenario)[-2]

    exp = exec_comms_range(base_scenario)
    logpath = "{path}/{title}.log".format(path=exp.exp_path,title=exp.title.replace(' ','_'))
    path = exp.exp_path

    print("Saved detection stats to {}".format(exp.exp_path))
    dump_trust_logs_and_stats_from_exp_paths([path], title=base_name)


