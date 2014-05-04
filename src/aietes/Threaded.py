#!/usr/bin/env python
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
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

__author__ = 'andrewbolster'

from multiprocessing import Process, JoinableQueue, cpu_count
from multiprocessing.process import current_process
import struct
import os
import logging
import gc

import numpy as np

from aietes import Simulation
from aietes.Tools import try_x_times


def sim_mask(args):
    # Properly Parallel
    #http://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    myid=current_process()._identity[0]
    np.random.seed(myid^struct.unpack("<L",os.urandom(4))[0])

    # Be Nice
    os.nice(5)

    lives=10
    kwargs, pp_defaults, retain_data = args
    sim_time = kwargs.pop("runtime", None)
    while True:
        try:
            sim = Simulation(**kwargs)
            logging.info("{} starting {}".format(current_process(), sim.title))
            prep_stats = sim.prepare(sim_time=sim_time)
            sim_time = sim.simulate()
            return_dict = sim.postProcess(**pp_defaults)
            if retain_data is True: #Implicitly implies boolean datatype
                return_val = sim.generateDataPackage()
            elif retain_data == "additional_only":
                dp = sim.generateDataPackage()
                return_val = dp.additional.copy()
            elif retain_data == "file":
                return_val = sim.generateDataPackage().write(kwargs.get("title"))
            else:
                return_val = return_dict
            del sim
            return return_val
        except RuntimeError:
            lives-=1
            if lives <= 0:
                raise
            else:
                logging.critical("{} died, restarting: {} lives remain".format(current_process(), lives ))
                del sim
        gc.collect()


def consumer(w_queue, r_queue):
    # Properly Parallel
    #http://stackoverflow.com/questions/444591/convert-a-string-of-bytes-into-an-int-python
    myid=current_process()._identity[0]
    np.random.seed(myid^struct.unpack("<L",os.urandom(4))[0])
    while True:
        try:
            uuid, simargs, postargs = w_queue.get()
            protected_run = try_x_times(5, RuntimeError,
                                        RuntimeError("Attempted two runs, both failed"),
                                        sim_mask)
            sim_results = protected_run((simargs, postargs))
        except Exception as e:
            sim_results = e
        finally:
            r_queue.put((uuid, sim_results))
            w_queue.task_done()
            print "Done %s" % uuid


def futures_version(arglist):
    import logging
    logging.basicConfig(level=logging.ERROR)
    from joblib import Parallel, delayed
    results = []
    try:
        results=Parallel(n_jobs=-1, verbose=10)(delayed(sim_mask)(args) for args in arglist)
    except Exception as e:
        logging.critical("Caught Exception: results is {}".format(len(results)))
        raise

    return results


work_queue = JoinableQueue()
result_queue = JoinableQueue()
cores = cpu_count()
running = False
workers = []


def boot():
    global running, workers
    workers = [Process(target=consumer, args=(work_queue, result_queue)) for i in range(cores)]
    for worker in workers:
        worker.start()
    print "started"
    running = True


def kill():
    global running, workers
    print "joining for death"
    work_queue.join()
    print "joined "
    for worker in workers:
        worker.terminate()
        del worker
    workers = []
    print "killed"
    running = False

