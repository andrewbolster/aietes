#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
import numpy

from polybos import ExperimentManager as ExpMan


def set_exp():
    e = ExpMan(node_count=8,
            title="Clumping Test", parallel=True)
    e.add_variable_range_scenario("clumping", numpy.linspace(0.0, 1.0, 20))
    return e


def run_exp(e):
    e.run(title="8-clumping-20",
          runcount=3,
          threaded=True)

    return e


if __name__ == "__main__":
    exp = set_exp()
    exp = run_exp(exp)
    ExpMan.print_stats(exp)
