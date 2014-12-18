#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
import numpy

from polybos import ExperimentManager as EXP


def set_exp():
    exp = EXP(node_count=8,
              title="Waypointing Test")
    exp.addVariable2RangeScenarios({"waypointing": numpy.linspace(0.0, 0.3, 20),
                                   "clumping": numpy.linspace(0.0, 0.3, 20)})
    return exp


def run_exp(exp):
    exp.run(title="8-wapointing-20_20-3",
            runcount=3,
            threaded=False)

    return exp


if __name__ == "__main__":
    exp = set_exp()
    exp = run_exp(exp)
    EXP.printStats(exp)
