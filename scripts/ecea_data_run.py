__author__ = 'bolster'

import ecea_basic_run, ecea_deltarange_run, ecea_nodecount_run, ecea_dvlrange_run
from datetime import datetime
import os

test_cases = [ecea_nodecount_run, ecea_deltarange_run, ecea_dvlrange_run]
#test_cases = [ecea_nodecount_run]

date = datetime.now().strftime('%Y%m%d-%H-%M')
for test in test_cases:
    print test.__name__
    exp = test.set_exp()
    exp.updateDuration(28800)
    try:
        exp.run(title="ECEA_Datarun_{}".format(date), no_time=True, runcount=32)
        exp.dump_dataruns()
    except RuntimeError as err:
        import traceback
        print("Experiment {} went horribly wrong, carrying on regardless and leaving a note: {}".format(test.__name__,
                                                                                                        err))
        traceback.print_exc(file=open(
            os.path.join(os.path.abspath(exp.exp_path),"FAILED.{}.log".format(test.__name__)),
            'w+'
            )
        )
