#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP


def set_exp():
    variations = [1, 0]
    e = EXP(node_count=4,
            title="FleetLawnmowerTOFVar-{}".format(variations),
            parallel=True, future=True,
            retain_data='files')
    e.updateDefaultNode({
        'behaviour': 'FleetLawnmower',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactor',
        'ecea': 'Simple2',
        'beacon_rate': 15
    })
    e.addVariableRangeScenario('tof_type', variations)
    e.updateDuration(28800)
    return e


def run_exp(e):
    e.run(title="ECEA_TOFRange_Test",
          runcount=16,
    )
    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







