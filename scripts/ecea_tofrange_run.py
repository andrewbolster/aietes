#!/usr/bin/env python
__author__ = 'andrewbolster'
from polybos import ExperimentManager as EXP

def set_exp():
    variations = [1, 0]
    exp = EXP(node_count=4,
              title="FleetLawnmowerTOFVar-{}".format(variations),
              parallel=True, future=True,
              retain_data='files')
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmower',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactor',
        'ecea':'Simple2',
        'beacon_rate':15
    })
    exp.addVariableRangeScenario('tof_type', variations)
    exp.updateDuration(28800)
    return exp


def run_exp(exp):
    exp.run(title="ECEA_TOFRange_Test",
            runcount=16,
            )
    return exp


if __name__ == "__main__":
    from aietes.Tools import memory
    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()







