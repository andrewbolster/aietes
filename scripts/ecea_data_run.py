# coding=utf-8
__author__ = 'bolster'

from datetime import datetime
import os
import ecea_deltarange_run
import ecea_nodecount_run
import ecea_dvlrange_run

test_cases = [ecea_nodecount_run, ecea_deltarange_run, ecea_dvlrange_run]
# test_cases = [ecea_nodecount_run]

date = datetime.now().strftime('%Y%m%d-%H-%M')
for test in test_cases:
    print test.__name__
    exp = test.set_exp()
    exp.update_duration(28800)
    try:
        exp.run(title="ECEA_Datarun_{0}".format(date), no_time=True, runcount=32)
        exp.dump_dataruns()
    except RuntimeError as err:
        import traceback

        print("Experiment {0} went horribly wrong, carrying on regardless and leaving a note: {1}".format(test.__name__,
                                                                                                        err))
        traceback.print_exc(file=open(
            os.path.join(os.path.abspath(exp.exp_path), "FAILED.{0}.log".format(test.__name__)),
            'w+'
        )
        )
