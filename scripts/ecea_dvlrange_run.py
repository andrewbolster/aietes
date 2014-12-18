#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    variations = [1e-3, 2e-3, 4e-3]
    exp = EXP(node_count=4,
              title="FleetLawnmowerDVLVar-{}".format(variations),
              parallel=True, future=True,
              retain_data='files')
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmower',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactor',
        'ecea':'Simple2',
        'beacon_rate':15
    })
    exp.addVariableRangeScenario('drift_dvl_scale', variations)
    exp.updateDuration(28800)
    return exp


def run_exp(exp):
    exp.run(title="ECEA_DVLRange_Test",
            runcount=32,
            )
    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







