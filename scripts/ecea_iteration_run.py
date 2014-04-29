#!/usr/bin/env python
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    exp = EXP(node_count=4,
              title="FleetLawnmower ECEA Model with varying filter iterations(1-8)",
              parallel=True, future=True,
              retain_data='files')
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmower',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactorPy',
    })
    exp.addVariableRangeScenario('ecea',["Simple{}".format(n) for n in range(1,8)])
    exp.updateDuration(21600)
    return exp

def run_exp(exp):
    exp.run(title="ECEA_Iteration_Test",
            runcount=64)

    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







