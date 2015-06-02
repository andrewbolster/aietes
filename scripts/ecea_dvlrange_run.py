#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP


def set_exp():
    variations = [1e-3, 2e-3, 4e-3]
    e = EXP(node_count=4,
            title="FleetLawnmowerDVLVar-{}".format(variations),
            parallel=True, future=True,
            retain_data='files')
    e.update_default_node({
        'behaviour': 'FleetLawnmower',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactor',
        'ecea': 'Simple2',
        'beacon_rate': 15
    })
    e.add_variable_range_scenario('drift_dvl_scale', variations)
    e.update_duration(28800)
    return e


def run_exp(e):
    e.run(title="ECEA_DVLRange_Test",
          runcount=32,
          )
    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







