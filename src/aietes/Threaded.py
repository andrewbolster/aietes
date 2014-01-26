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
import struct, os

import futures

import numpy as np

from aietes import Simulation
from aietes.Tools import try_x_times


def sim_mask(args):
    kwargs, pp_defaults = args
    sim_time = kwargs.pop("runtime", None)
    sim = Simulation(**kwargs)
    prep_stats = sim.prepare(sim_time=sim_time)
    sim_time = sim.simulate()
    return_dict = sim.postProcess(**pp_defaults)
    print("%s(%s):%f%%"
          % (current_process().name, return_dict.get('data_file', "N/A"),
             100.0 * float(sim_time) / prep_stats['sim_time']))
    return sim.generateDataPackage()


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
    results = [None] * len(arglist)
    with futures.ProcessPoolExecutor() as exe:
        for i, (kwargs, pp_args) in enumerate(arglist):
            e = exe.submit(sim_mask, (kwargs, pp_args))
            results[i] = e.result()
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
    print "joining"
    work_queue.join()
    print "joined"
    for worker in workers:
        worker.terminate()
        del worker
    workers = []
    print "killed"
    running = False

