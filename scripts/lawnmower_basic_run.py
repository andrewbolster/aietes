#!/usr/bin/env python
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    exp = EXP(node_count=4,
              title="LawnmowerWithDriftOnly",
              parallel=True, future=True,
              retain_data='files')
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmowerLoop',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactorPy',
    })
    exp.addDefaultScenario()
    exp.updateDuration(21600)
    return exp

def run_exp(exp):
    exp.run(title="Lawnmower_Basic_Test",
            runcount=8)

    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







