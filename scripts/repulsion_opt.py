#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
import numpy

from polybos import ExperimentManager as ExpMan


def set_exp():
    e = ExpMan(node_count=8,
            title="Repulsion-Clumping Test")
    e.add_variable_2_range_scenarios({"repulsion": numpy.linspace(0.005, 0.3, 10),
                                      "clumping": numpy.linspace(0.005, 0.3, 10)})
    return e


def run_exp(e):
    e.run(title="8-repulsion-10_10-3",
          runcount=3,
          threaded=False)

    return e


if __name__ == "__main__":
    exp = set_exp()
    exp = run_exp(exp)
    ExpMan.print_stats(exp)
