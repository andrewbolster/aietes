#!/usr/bin/env python
# coding=utf-8
"""
 * This file is part of the Aietes Framework (https://github.com/andrewbolster/aietes)
 *
 * (C) Copyright 2013 Andrew Bolster (http://andrewbolster.info/) and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

from multiprocessing.process import current_process
from multiprocessing import Pool
import struct
import os
import logging
import gc
from time import gmtime, strftime

from joblib import Parallel, delayed
import numpy as np

from aietes import Simulation


def sim_mask(args):
    # Properly Parallel
    # http://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    myid = current_process()._identity[0]
    np.random.seed(myid ^ struct.unpack("<L", os.urandom(4))[0])

    # Be Nice
    niceness = os.nice(0)
    os.nice(5 - niceness)

    lives = 10
    kwargs, pp_defaults, retain_data = args
    sim_time = kwargs.pop("sim_time", None)
    while True:
        try:
            sim = Simulation(**kwargs)
            logging.info("{} starting {}".format(current_process(), sim.title))
            prep_stats = sim.prepare(sim_time=sim_time)
            sim_time = sim.simulate()
            return_dict = sim.postprocess(**pp_defaults)
            if retain_data is True:  # Implicitly implies boolean datatype
                return_val = sim.generate_datapackage()
            elif retain_data == "additional_only":
                dp = sim.generate_datapackage()
                return_val = dp.additional.copy()
            elif retain_data == "file":
                return_val = sim.generate_datapackage().write(
                    kwargs.get("title"))
            else:
                return_val = return_dict
            del sim
            return return_val
        except (KeyboardInterrupt, SystemExit):
            raise
        except RuntimeError:
            lives -= 1
            if lives <= 0:
                raise
            else:
                logging.critical(
                    "{} died, restarting: {} lives remain".format(current_process(), lives))
                del sim
        gc.collect()


def queue_mask(i, args):
    try:
        results = sim_mask(args)
        return i, results
    except:
        return


def parallel_sim(arglist):
    import logging

    logging.basicConfig(level=logging.DEBUG)

    results = []
    print "Beginning Parallel Run of {runcount} at {t}".format(
        runcount=len(arglist), t=strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    )
    try:
        results = Parallel(
            n_jobs=-1, verbose=10)(delayed(sim_mask)(args) for args in arglist)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logging.critical(
            "Caught Exception: results is {}".format(len(results)))
        raise

    return results


class queue_sim(object):
    def __init__(self, arglist, pool):
        import logging

        logging.basicConfig(level=logging.DEBUG)
        self.tasklist = arglist
        self.results = [None] * len(self.tasklist)
        self.pending_results = [None] * len(self.tasklist)
        self.pool = pool

    def launch(self):
        logging.info("Appending Queued Parallel Run of {runcount} at {t}".format(
            runcount=len(self.tasklist), t=strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        ))
        for i, args in enumerate(self.tasklist):
            self.pending_results[i] = self.pool.apply_async(queue_mask, args=(i, args))

    def finished(self):
        """
        Is execution complete? (Note, this may be different than when results are ready)
        :return: bool
        """
        return all([r.ready() for r in self.pending_results])

    def populate(self):
        """
        Populate the results list from the pending list.
        Should be blocking after close
        Unknown behaviour if mask fails.
        :return: bool: results are ready for processing
        """
        if self.finished():
            for i,p in enumerate(self.pending_results):
                # If the task raised an exception, it will appear here
                _id, _data = p.get()
                if _id != i:
                    raise AssertionError("Results should have been queued and processed in the same order, instead I've got id {} when I'm expecting {}".format(
                        _id, i
                    ))
                self.results[i] = _data
            return True
        else:
            return False


