__author__ = 'bolster'

import ecea_basic_run, ecea_deltarange_run, ecea_nodecount_run
from contrib.Ghia.ecea.data_grapher import data_grapher
from datetime import datetime

test_cases = [ecea_basic_run, ecea_nodecount_run, ecea_deltarange_run]
#test_cases = [ecea_nodecount_run]

date = datetime.now().strftime('%Y%m%d-%H-%M')
for test in test_cases:
    print test.__name__
    exp = test.set_exp()
    exp.updateDuration(28800)
    exp.run(title="ECEA_Datarun_{}".format(date), no_time=True, runcount=32)
    exp.dump_dataruns()
    data_grapher(dir=exp.exp_path,title=exp.title)

