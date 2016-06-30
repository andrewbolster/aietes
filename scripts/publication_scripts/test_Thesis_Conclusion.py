import unittest
import pandas as pd
import os
from IPython.display import display, Latex
from bounos.Analyses.Trust import generate_node_trust_perspective
from bounos.Analyses.Dixon import *
from aietes.Tools import *
from aietes.Tools import _results_dir
import seaborn as sns
from sklearn_pandas import DataFrameMapper, cross_val_score
import sklearn.preprocessing, sklearn.decomposition, \
      sklearn.linear_model, sklearn.pipeline, sklearn.metrics
import matplotlib.pyplot as plt

from scripts.publication_scripts import group_dixon_test

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"
fig_path = "/home/bolster/src/thesis/Figures/"

# Once the above is done, should be able to make averages across target behaviours
def _detick(x):
    if isinstance(x, (float, np.float)):
        return x
    elif x.startswith('\\'):
        return True
    else:
        return False
class MultiDomainIdentification(unittest.TestCase):

    culprit='Alfa'

    fig_path = "/home/bolster/src/thesis/Figures/"
    fuggit=False
    @classmethod
    def setUpClass(cls):
        try:
            if cls.fuggit:
                raise KeyError("FUCK YOU")
            with pd.get_store(os.path.join(results_path, 'multidomainidentification.h5')) as s:
                print(s.keys())
                cls.dixon_df=s.get('dixon_df')
                cls.w_dfs=s.get('w_df')
                cls.synth_weights=s.get('synth_weights')
                cls.w_pt=s.get('w_pt')
                important_done = True
                print("Found important stuff")
        except KeyError:
            print("Must regenerate important stuff")
            important_done = False

        if not important_done:
            globs = filter(lambda s: s.startswith('Global') and s.endswith('.h5'), os.listdir(_results_dir))
            t_acc = None
            for g in globs:
                with pd.get_store(os.path.join(_results_dir, g)) as s:
                    t = s.get('trust').dropna()
                    if t_acc is None:
                        t_acc = t
                    else:
                        _tinter = t.reset_index()
                        _tinter['run'] += t_acc.index.get_level_values('run').max() + 1
                        t_acc = t_acc.append(_tinter.set_index(t_acc.index.names))
            t_acc.shape
            trust_observations = t_acc
            map_levels(trust_observations, var_rename_dict)

            with pd.get_store(os.path.join(results_path, 'w_df.h5')) as s:
                w_df = s.get('weights_381')
                w_df_orig = s.get('weights_5')
                w_df_orig.index.rename(['Target', 'Subject'], inplace=True)

            with pd.get_store('/dev/shm/top_dt.h5') as s:
                df = pd.concat([s.get(k) for k in s.keys()], keys=[k[1:] for k in s.keys()], names=['Target Behaviour', ''])
            df = df.applymap(_detick).groupby(level='Target Behaviour').head(1).groupby(level='Target Behaviour').mean()
            _t = df.ix['Mean']
            df = df.drop('Mean', axis=0).append(_t)

            synth_weights = pd.concat([g.reset_index('subset', drop=True)
                                       for my_domain, my_domain_subset in df['Metrics in Synthetic Domain'].iterrows()
                                       for k, g in w_df.groupby(level='subset')
                                       if my_domain_subset.equals(k.T)
                                       ],
                                      keys=df['Metrics in Synthetic Domain'].index,
                                      names=['Target', 'Subject'])
            #   synth_weights=w_df_orig.append(synth_weights)
            synth_weights.rename(columns=invert_dict(metric_rename_dict), inplace=True)

            w_trust_m = {}
            for m in synth_weights.index.get_level_values('Target').unique():
                if m != 'Mean':
                    w_trust_m[m] = generate_node_trust_perspective(trust_observations,
                                                                   metric_weights=synth_weights.xs((m, m)))
            w_trust_df = pd.concat(w_trust_m.values(), keys=w_trust_m.keys(),
                                   names=['target_domain', 'var', 'run', 'observer', 't'])
            w_trust_df.columns.names = w_trust_df.columns.names[:-1] + ['node']
            w_trust_df = w_trust_df.stack('node').reset_index()
            w_trust_df.columns = w_trust_df.columns[:-1].tolist() + ['trust']

            w_pt = pd.pivot_table(w_trust_df, index=['var', 'run', 'observer', 'node', 't'],
                                  columns='target_domain', values='trust')

            def _ewma(gf):
                return gf.ewm(alpha=0.5).mean()


            cls.w_pt = w_pt.groupby(level=['var', 'run', 'observer']).apply(_ewma)

            sf = pd.concat([
                               pd.concat(
                                   [group_dixon_test(g, gf)
                                    for g, gf in cls.w_pt.xs(var, level='var', drop_level=False).groupby(
                                       level=['var', 'run', 'node']).sum().groupby(level=['var', 'run'])
                                    ])
                               for var in cls.w_pt.index.get_level_values('var').unique()
                               ]
                           )
            sf.columns = ['Behaviour', 'Run', 'Target Domain', 'Suspect']

            sf['True Positive'] = (sf['Suspect'] == cls.culprit) != ((sf['Behaviour'] == "Fair") & (sf['Suspect'] == 'None')).astype(float)
            sf['False Positive'] = (sf['Suspect'] != 'None') & ((sf['Behaviour'] == "Fair") | (sf['Suspect'] != cls.culprit)).astype(float)

            cls.dixon_df = sf
            cls.w_df = w_df
            cls.synth_weights = synth_weights

            important = dict(
                dixon_df=cls.dixon_df,
                w_df=cls.w_df,
                synth_weights=cls.synth_weights,
                w_pt = cls.w_pt
            )

            for k,v in important.iteritems():
                v.to_hdf(os.path.join(results_path, 'multidomainidentification.h5'), k)

    def test_launch(self):
        self.assertTrue(True)

    def test_confidence_table(self):
        intergfs = []
        for g, gf in self.w_pt.groupby(level=['var', 'run', 'node']).sum().groupby(level=['var', 'run']):
            _intergf = gf.unstack('node').stack('target_domain')
            intergfs.append(_intergf[self.culprit] / _intergf.drop(self.culprit, axis=1).mean(axis=1))

        confidences = pd.concat(intergfs).unstack('target_domain')
        c_table = confidences.groupby(level='var').describe().unstack('var').iloc[[1, 2]].stack('var')
        c_table.columns.name = 'Domain'
        c_table.index.names = ['', 'Behaviour']
        with open(os.path.join(self.fig_path,"confidence_multi.tex"), 'w') as f:
            c_table.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))


    def test_overall_stats(self):
        tp = self.dixon_df
        overall_stats = self.dixon_df.groupby('Behaviour')['True Positive'].describe().unstack('Behaviour').iloc[[1, 2]].T
        overall_stats.columns = ['Mean', 'Std']
        overall_stats.index.name = 'Behaviour'
        with open(os.path.join(self.fig_path, "overall_multi_stats.tex"), 'w') as f:
            overall_stats.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))


    def test_per_metric_stats(self):
        per_metric_stats = self.dixon_df.groupby(['Behaviour', 'Target Domain']).describe().unstack(['Behaviour', 'Target Domain']).iloc[
            [1, 2]].stack('Behaviour')
        per_metric_stats.columns.name = 'Domain'
        per_metric_stats.index.names = ['', 'Behaviour']
        with open(os.path.join(self.fig_path, "per_metric_multi_stats.tex"), 'w') as f:
            per_metric_stats.to_latex(buf=f, float_format=lambda f: "{0:1.3f}".format(f))


if __name__ == '__main__':
    unittest.main()
