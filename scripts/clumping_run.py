#!/usr/bin/env python
__author__ = 'andrewbolster'
import numpy

from polybos import ExperimentManager as EXP


def set_exp():
    exp = EXP(node_count=8,
              title="Clumping Test")
    exp.addVariableRangeScenario("clumping", numpy.linspace(0.0, 1.0, 20))
    return exp


def run_exp(exp):
    exp.run(title="8-clumping-20",
            runcount=3,
            threaded=False)

    return exp


def run_suite(exp):
    for s in exp.scenarios:
        stats = s.generateRunStats()
        print(s.title)
        print("\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)" % (
            avg_of_dict(stats, ['motion', 'fleet_distance']), avg_of_dict(stats, ['motion', 'fleet_efficiency']),
            avg_of_dict(stats, ['motion', 'std_of_INDA']), avg_of_dict(stats, ['motion', 'std_of_INDD']),
            avg_of_dict(stats, ['achievements', 'max_ach']), avg_of_dict(stats, ['achievements', 'avg_completion']) * 100.0
        )
        )

        for r in stats:
            print("\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)" % (
                r['motion']['fleet_distance'], r['motion']['fleet_efficiency'],
                r['motion']['std_of_INDA'], r['motion']['std_of_INDD'],
                r['achievements']['max_ach'], r['achievements']['avg_completion'] * 100.0
            )
            )


def avg_of_dict(dict_list, keys):
    sum = 0
    count = 0
    for d in dict_list:
        count += 1
        for key in keys[:-1]:
            d = d.get(key)
        sum += d[keys[-1]]
    return float(sum) / count


if __name__ == "__main__":
    exp = set_exp()
    exp = run_exp(exp)
    run_suite(exp)