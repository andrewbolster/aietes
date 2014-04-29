#!/usr/bin/env python
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    variations=[2,4,8,9,12,16,18]
    exp = EXP(node_count=4,
              title="FleetLawnmower ECEA Model with varying node counts ({})".format(variations),
              parallel=True, future=True,
              retain_data='files')
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmowerLoop',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactorPy',
        'ecea':'Simple2'
    })
    exp.addVariableNodeScenario(variations)
    exp.updateDuration(21600)
    return exp


def run_exp(exp):
    exp.run(title="ECEA_NodeCount_Test",
            runcount=8,
            )
    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







