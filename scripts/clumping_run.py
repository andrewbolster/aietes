#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
import numpy

from polybos import ExperimentManager as EXP


def set_exp():
    exp = EXP(node_count=8,
              title="Clumping Test", parallel=True)
    exp.addVariableRangeScenario("clumping", numpy.linspace(0.0, 1.0, 20))
    return exp


def run_exp(exp):
    exp.run(title="8-clumping-20",
            runcount=3,
            threaded=True)

    return exp


if __name__ == "__main__":
    exp = set_exp()
    exp = run_exp(exp)
    EXP.printStats(exp)
