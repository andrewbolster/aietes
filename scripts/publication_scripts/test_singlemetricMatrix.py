
import os
import pandas as pd


class EXPShadow(object):
    @property
    def exp_path(self):
        return "/home/bolster/src/aietes/results/Malicious Behaviour Trust Control-2016-02-25-10-41-55"

exp = EXPShadow()
print exp.exp_path


    print os.path.join(path, "outliers.h5")

if __name__ == '__main__':
    generate_outliers(exp.exp_path, runs=4)
    observer = 'Bravo'
    target = 'Alfa'
    n_metrics = 9
    outlier_weights = Weight.build_outlier_weights(os.path.join(exp.exp_path, "outliers.h5"), observer=observer, target=target, n_metrics=n_metrics, signed=False)
