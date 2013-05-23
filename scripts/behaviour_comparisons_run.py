__author__ = 'andrewbolster'

from polybos import ExperimentManager as EXP


def set():
    exp = EXP(node_count=8,
              title="Malicious Behaviour Trust Comparison")
    exp.addVariableAttackerBehaviourSuite(["Waypoint", "Shadow", "SlowCoach"], n_attackers=1)
    return exp


def run(exp):
    exp.run(title="8-bev-mal",
            runcount=3,
            runtime=5000)
    return exp


def stats(exp):
    print("Run\tFleet D, Efficiency\tstd(INDA,INDD)\tAch., Completion Rate\t")
    for s in exp.scenarios:
        stats = s.generateRunStats()
        print("%s,%s" % (s.title, [(bev, nodelist)
                                   for bev, nodelist in s.getBehaviourDict().iteritems()
                                   if '__default__' not in nodelist
        ]))
        print("AVG\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)" % (
            avg_of_dict(stats, ['motion', 'fleet_distance']), avg_of_dict(stats, ['motion', 'fleet_efficiency']),
            avg_of_dict(stats, ['motion', 'std_of_INDA']), avg_of_dict(stats, ['motion', 'std_of_INDD']),
            avg_of_dict(stats, ['achievements', 'max_ach']),
            avg_of_dict(stats, ['achievements', 'avg_completion']) * 100.0
        )
        )

        for i, r in enumerate(stats):
            print("%d\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)" % (
                i,
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


def set_run():
    return run(set())


if __name__ == "__main__":
    exp = set()
    exp = run(exp)
    stats(exp)