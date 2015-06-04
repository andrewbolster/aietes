#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as ExpMan


def set_exp():
    e = ExpMan(node_count=4,
               title="FleetLawnmowerBasicRun",
               parallel=True, future=True,
               retain_data='files')
    e.update_default_node({
        'behaviour': 'FleetLawnmower',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactor',
        'ecea': 'Simple2'
    })
    e.add_default_scenario()
    e.update_duration(21600)
    return e


def run_exp(e):
    e.run(title="ECEA_Basic_Test",
          runcount=16)

    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()
