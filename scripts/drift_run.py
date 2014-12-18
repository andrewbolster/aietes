#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    exp = EXP(node_count=4,
              title="Drift Analysis",
              parallel=True, future=True,
              retain_data=False)
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmower',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactorPy'
    })
    exp.addDefaultScenario()
    exp.updateDuration(28800)
    return exp


def run_exp(exp):
    exp.run(title="DriftAnalysis",
            runcount=250,
            )
    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print memory()







