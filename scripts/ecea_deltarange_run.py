#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    variations = [1,5,15,30,60,120,300,600]
    exp = EXP(node_count=4,
              title="FleetLawnmowerBeaconVar-{}".format(variations),
              parallel=True, future=True,
              retain_data='files')
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmower',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactorPy',
        'ecea':'Simple2'
    })
    exp.addVariableRangeScenario('beacon_rate', variations)
    exp.updateDuration(21600)
    return exp


def run_exp(exp):
    exp.run(title="ECEA_DeltaRange_Test",
            runcount=64,
            )
    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







