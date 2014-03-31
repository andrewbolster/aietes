__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP
from bounos import Analyses, _metrics

from contextlib import contextmanager
import sys

@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'a')
    yield
    sys.stdout = saved_stdout

def set():
    exp = EXP(node_count=8,
              title="Malicious Behaviour Trust Comparison",
              parallel=True,
              future=True
             )
    exp.addVariableAttackerBehaviourSuite(["Waypoint", "Shadow", "SlowCoach"], n_attackers=1)
    return exp


def run(exp):
    exp.run(title="8-bev-mal-64r-2000t",
            runcount=16,
            runtime=8000,
            dataFile=True)
    return exp


def set_run():
    return run(set())

if __name__ == "__main__":
    exp = set()
    exp = run(exp)
    exp.dump_analysis()
    with redirected(stdout="%s.log"%exp.title):
        EXP.printStats(exp)
