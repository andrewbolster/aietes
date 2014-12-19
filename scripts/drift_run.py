#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP


def set_exp():
    e = EXP(node_count=4,
            title="Drift Analysis",
            parallel=True, future=True,
            retain_data=False)
    e.update_default_node({
        'behaviour': 'FleetLawnmower',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactorPy'
    })
    e.add_default_scenario()
    e.update_duration(28800)
    return e


def run_exp(e):
    e.run(title="DriftAnalysis",
          runcount=250,
    )
    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print memory()







