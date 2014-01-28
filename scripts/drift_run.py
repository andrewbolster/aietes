#!/usr/bin/env python
__author__ = 'andrewbolster'
import numpy, os
from matplotlib import pyplot as plt
from polybos import ExperimentManager as EXP


def set_exp():
    exp = EXP(node_count=4,
              title="Drift Analysis",
              parallel=True, future=True,dataFile=False)
    exp.updateDefaultNode({
        'behaviour':'FleetLawnmower',
        'waypoint_style':'lawnmower',
        'positioning':'surface',
        'drifting':'DriftFactorPy'
    })
    exp.updateEnvironment([])
    exp.useDefaultScenario()
    exp.updateDuration(28800)
    return exp


def run_exp(exp):
    exp.run(title="DriftAnalysis",
            runcount=100,
            )
    try:
        exp.dump_dataruns()
    except:
        print("DIED ON THE BOGS BUT WILL CONTINUE!")


    return exp


if __name__ == "__main__":
    exp = set_exp()
    exp = run_exp(exp)
    # N for the purpose of RMS calc is nodes * runs
    n = exp.runcount*exp.node_count
    scenario=exp.scenarios[0]
    sq_error = numpy.power([dp.drift_error() for dp in scenario.datarun], 2)
    rms = numpy.sqrt(numpy.sum(sq_error,axis=(0,1))/n)
    f=plt.gcf()
    title='RMS characteristics for {} runs'.format(n)
    ax=f.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('mission time(s)')
    ax.set_ylabel('RMS drift(m^-2)')
    plt.plot(rms)
    f.savefig(os.path.join(exp.exp_path,title))




