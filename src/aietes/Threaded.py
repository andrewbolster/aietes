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

import threading
from Queue import Queue

from aietes import Simulation


class SimulationThread(threading.Thread):
    def __init__(self, runcount=1, sim_args=None, pp_args=None, **kwargs):
        self.runcount = runcount
        self.sim_args = sim_args
        self.pp_args = pp_args
        threading.Thread.__init__(self)
        self.sim = Simulation(**self.sim_args)
        self.prep_stats = self.sim.prepare()

    def run(self):
        try:
            self.sim_stats = self.sim.simulate()
            self.sim.postProcess(**self.pp_args)
        except Exception as exp:
            raise

    def get_result(self):
        return self.runcount, self.sim.generateDataPackage()


def producer(q, runcount, sim_args=None, pp_args=None):
    base_title = sim_args['title']
    for run in range(runcount):
        try:
            sim_args.update({'title': base_title + "-%s" % run})
            sim = SimulationThread(sim_args=sim_args, pp_args=pp_args, runcount=run)
            sim.start()
        except Exception as exp:
            raise


finished = []


def consumer(q, count):
    while len(finished) < count:
        thread = q.get(True)
    thread.join()
    finished.append(thread.get_result())


def go(runcount, sim_args=None, pp_args=None):
    sim_queue = Queue(4)
    p_thread = threading.Thread(target=producer, args=(sim_queue, runcount, sim_args, pp_args))
    c_thread = threading.Thread(target=consumer, args=(sim_queue, runcount))
    p_thread.start()
    c_thread.start()
    p_thread.join()
    c_thread.join()
    dataset = [None for _ in range(runcount)]
    for res_tup in finished:
        run, result = res_tup
        dataset[run] = result
    return dataset



