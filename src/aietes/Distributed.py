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
 *     Andrew Bolster, Queen's University Belfast (-July 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import logging
from time import gmtime, strftime
from celery import Celery
from aietes.Threaded import sim_mask

app = Celery('aietes.Distributed', broker='amqp://guest@localhost//', backend='amqp')

@app.task
def simulate(i, args):
    try:
        results = sim_mask(args)
        return i,results
    except Exception as e:
        return i, e


class emmitter(object):
    def __init__(self, arglist):

        logging.basicConfig(level=logging.DEBUG)
        self.tasklist = arglist
        self.results = None
        self.pending_results = [None]*len(self.tasklist)

    def launch(self):
        logging.info("Appending Celery Run of {runcount} at {t}".format(
            runcount=len(self.tasklist), t=strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        ))
        for i,args in enumerate(self.tasklist):
            logging.info('Dispatching {}'.format(i))
            self.pending_results[i]=simulate.delay(i,args)

    def finished(self):
        if all([r.ready() for r in self.pending_results]):
            if self.results is None:
                self.results = [r.get() for r in self.pending_results]
            return True
        else:
            return False
