# coding=utf-8
import pandas as pd

from bounos.Analyses.Dixon import dixon_test_95
from .helpers import *


def group_dixon_test(g,gf):
    acc = []
    for k,v in gf.T.apply(dixon_test_95, axis=1).T.iteritems():
        outlier = map(lambda x:x[-1],gf.loc[gf[k]==v].index.tolist())
        if outlier:
            acc.append((g[0],g[1],k,outlier[0]))
        else:
            acc.append((g[0],g[1],k,'None'))

        if len(outlier)>1:
            raise ValueError("Haven't written the case for multiple outliers yet")
    return pd.DataFrame(acc, columns=['var','run','metric','target'])