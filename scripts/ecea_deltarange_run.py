#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as ExpMan


def set_exp():
    variations = [1, 5, 15, 30, 60, 120, 300, 600]
    e = ExpMan(node_count=4,
               title="FleetLawnmowerBeaconVar-{}".format(variations),
               parallel=True, future=True,
               retain_data='files')
    e.update_default_node({
        'behaviour': 'FleetLawnmower',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactorPy',
        'ecea': 'Simple2'
    })
    e.add_custom_node_scenario('beacon_rate', variations)
    e.update_duration(21600)
    return e


def run_exp(e):
    e.run(title="ECEA_DeltaRange_Test",
          runcount=64,
          )
    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()
