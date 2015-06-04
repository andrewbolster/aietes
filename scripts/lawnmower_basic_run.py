#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as ExpMan


def set_exp():
    e = ExpMan(node_count=4,
               title="LawnmowerWithDriftOnly",
               parallel=True, future=True,
               retain_data='files')
    e.update_default_node({
        'behaviour': 'FleetLawnmowerLoop',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactorPy',
    })
    e.add_default_scenario()
    e.update_duration(21600)
    return e


def run_exp(e):
    e.run(title="Lawnmower_Basic_Test",
          runcount=8)

    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()
