
import os
import pandas as pd
import bounos.Analyses.Weight as Weight

class EXPShadow(object):
    @property
    def exp_path(self):
        return "/home/bolster/src/aietes/results/Malicious Behaviour Trust Control-2016-02-24-14-40-20"

exp = EXPShadow()
print exp.exp_path


for run in range(4):
    with pd.get_store(exp.exp_path + '.h5') as store:
        #sub_frame = pd.concat([
        #    store.trust.xs('Alfa', level='observer', drop_level=False),
        #    store.trust.xs('Bravo', level='observer', drop_level=False),
        #    store.trust.xs('Charlie', level='observer', drop_level=False)
        #]).xs(run, level='run', drop_level=False)
        sub_frame = store.trust

    outliers = Weight.perform_weight_factor_outlier_analysis_on_trust_frame(sub_frame, "CombinedTrust", extra=run,
                                                                         min_emphasis=0,
                                                                         max_emphasis=2, max_sum=1, par=True)
    outliers.to_hdf(os.path.join(exp.exp_path, "outliers.h5"), "CombinedTrust_{}_4".format(run))
