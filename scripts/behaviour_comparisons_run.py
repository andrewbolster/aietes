__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP
from bounos import Analyses, _metrics


def set():
    exp = EXP(node_count=8,
              title="Malicious Behaviour Trust Comparison",
             )
    exp.addVariableAttackerBehaviourSuite(["Waypoint", "Shadow", "SlowCoach"], n_attackers=1)
    return exp


def run(exp):
    exp.run(title="8-bev-mal",
            runcount=50,
            runtime=2000)
    return exp


def set_run():
    return run(set())

if __name__ == "__main__":
    exp = set()
    exp = run(exp)
    EXP.printStats(exp)
