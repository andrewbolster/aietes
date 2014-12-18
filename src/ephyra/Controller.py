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
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import functools
import logging
import os

from joblib import Parallel, delayed


parallel = False  # Doesn't make a damned difference.
if parallel:
    os.system("taskset -p 0xff %d" % os.getpid())

from bounos import BounosModel
from bounos.Metrics import *
from aietes.Tools import itersubclasses, timeit


def log_and_call():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # set name_override to func.__name__
            logging.info("Entered %s (%s)" % (func.__name__, kwargs))
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_model():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            assert self.model_is_ready(), "Model not properly instantiated"
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class EphyraController(object):
    def __init__(self, *args, **kw):
        self.model = BounosModel()
        self.view = None
        self._metrics_availiable = list(itersubclasses(Metric))
        self._metrics_enabled = [metric for metric in self._metrics_availiable
                                 if not getattr(metric, 'drift_enabled', False) and not getattr(metric, 'ecea_enabled', False)]
        self.metrics = []
        self.args = kw.get("exec_args", None)

        if self.args is not None and self.args.data_file is not None:
            self.load_data_file(self.args.data_file)

    def load_data_file(self, file_path):
        """
        Raises IOError on File not found
        """
        self.model.import_datafile(file_path)
        logging.debug("Loaded Datafile from {}".format(file_path))
        self.update_metrics()

    def model_is_ready(self):
        return self.model.is_ready

    def is_simulation(self):
        return self.model.is_simulating

    def set_model(self, model):
        self.model = model
        self.update_metrics()

    @check_model()
    def update_model(self, *args, **kw):
        # Update Base Data
        self.model.update_data_from_sim(*args, **kw)
        self.update_metrics()

    @check_model()
    @timeit()
    def update_metrics(self):
        # Update secondary metrics
        if self.drifting():
            for drift_metric in [m for m in self._metrics_availiable
                                 if getattr(m, 'drift_enabled', False) and m not in self._metrics_enabled]:
                logging.info(
                    "Adding {} to metrics as data is Drift-Enabled".format(drift_metric))
                self._metrics_enabled.append(drift_metric)
        if self.ecea():
            for ecea_metric in [m for m in self._metrics_availiable
                                if getattr(m, 'ecea_enabled', False) and m not in self._metrics_enabled]:
                logging.info(
                    "Adding {} to metrics as data is ECEA-Enabled".format(ecea_metric))
                self._metrics_enabled.append(ecea_metric)
        if not len(getattr(self, "metrics", [])) == len(self._metrics_enabled):
            self.rebuild_metrics()

        if parallel:
            n_jobs = -1
        else:
            n_jobs = 1
        self.metrics = Parallel(n_jobs=n_jobs, verbose=30, max_nbytes='1M')(
            delayed(metric)(self.model) for metric in self.metrics)

    @check_model()
    @timeit()
    def rebuild_metrics(self):
        # in the case of metrics_enabled being changed, this requires a
        # complete rebuild
        logging.info("Rebuilding Metrics:{}".format(self._metrics_enabled))
        self.metrics = map(lambda m: m(), self._metrics_enabled)
        return self.metrics

    @check_model()
    def get_metrics(self, i=None, *args, **kw):
        if i is None:
            return self.metrics
        elif isinstance(i, int):
            return self.metrics[i]
        else:
            raise NotImplementedError(
                "Metrics Views must be addressed by int's or nothing! (%s)" % str(i))

    def set_view(self, view):
        self.view = view

    def run_simulation(self, config):
        # TODO will raise ConfigError on, well, config errors
        pass

    @check_model()
    def get_raw_positions_log(self):
        return self.model.p

    @check_model()
    def get_vector_names(self, i=None):
        """

        :param i:
        :return:
        """
        return self.model.names if i is None else self.model.names[i]

    def get_n_vectors(self):
        """


        :return:
        """
        try:
            return self.model.n
        except AttributeError:
            return 0

    @check_model()
    def get_model_title(self):
        """


        :return:
        """
        return self.model.title

    @check_model()
    def get_extent(self):
        """


        :return:
        """
        return self.model.environment

    @check_model()
    def get_final_tmax(self):
        """
        Returns the highest addressable time index that the dataset WILL occupy
        """
        if self.model.is_simulating:
            return self.model.simulation.tmax - 1
        else:
            return self.model.tmax - 1

    @check_model()
    def get_3D_trail(self, node=None, time_start=None, length=None):
        """



        :param node:
        :param time_start:
        :param length:
        Return the [X:][Y:][Z:] trail for a given node from time_start backwards to
        a given length

        If no time given, assume the full time range
        """
        time_start = time_start if time_start is not None else self.get_final_tmax()
        time_end = max(0 if length is None else (time_start - length), 0)

        if node is None:
            return self.model.p[:, :, time_start:time_end:-1].swapaxes(0, 1)
        else:
            return self.model.p[node, :, time_start:time_end:-1]

    @check_model()
    def get_fleet_positions(self, time):
        """

        :param time:
        :return:
        """
        return self.model.position_slice(time)

    def get_fleet_headings(self, time):
        """

        :param time:
        :return:
        """
        return self.model.heading_slice(time)

    def get_node_contribs(self, node, time=None):
        """

        :param node:
        :param time:
        :return:
        """
        return self.model.contribution_slice(node, time)

    def get_max_node_contribs(self):
        """
        Used to get consistent colour maps for behaviours
        """
        return max(len(c) for c in self.model.contributions[:, 0])

    def get_contrib_keys(self):
        """


        :return:
        """
        return self.model.contributions[np.argmax(len(self.model.contributions[:, 0])), 0].keys()

    def get_achievements(self):
        """


        :return:
        """
        if self.model.achievements is not None:
            return self.model.achievements.nonzero()[1]
        else:
            return None

    def get_fleet_average_pos(self, time):
        """

        :param time:
        :return:
        """
        return np.average(self.model.position_slice(time), axis=0)

    @check_model()
    def get_fleet_configuration(self, time):
        """
           Returns a dict of the fleet configuration:

           :param time: time index to calculate at
           :type time int

           :returns dict {positions, headings, avg_pos, avg_head, stdev_pos, stdev_head}
           """
        positions = self.get_fleet_positions(time)
        avg_pos = self.get_fleet_average_pos(time)
        _distances_from_avg_pos = map(lambda v: mag(v - avg_pos), positions)
        stddev_pos = np.std(_distances_from_avg_pos)
        headings = self.get_fleet_headings(time)
        avg_head = np.average(headings, axis=0)
        _distances_from_avg_head = map(lambda v: mag(v - avg_head), headings)
        stddev_head = np.std(_distances_from_avg_head)

        return dict({'positions':
                         {
                             'pernode': positions,
                             'avg': avg_pos,
                             'stddev': stddev_pos,
                             'delta_avg': _distances_from_avg_pos
                         },
                     'headings':
                         {
                             'pernode': headings,
                             'avg': avg_head,
                             'stddev': stddev_head,
                             'delta_avg': _distances_from_avg_head
                         }
        })

    def get_heading_mag_max_min(self):
        """


        :return:
        """
        mags = self.model.heading_mag_range()
        return max(mags), min(mags)

    def get_position_min_max(self, time):
        """
        Return a 3-tuple of (min,max),(min,max),(min,max) for x,y,z respectively for a given time
        :param time:
        """
        pos = self.get_fleet_positions(time)
        mins = pos.min(axis=0)
        maxes = pos.max(axis=0)
        return tuple(zip(mins, maxes))

    def get_position_stddev_max_min(self):
        """


        :return:
        """
        stddevs = self.model.distance_from_average_stddev_range()
        return max(stddevs), min(stddevs)

    @check_model()
    def get_waypoints(self):
        """


        :return:
        """
        wp = getattr(self.model, 'waypoints', None)
        if isinstance(wp, np.ndarray) and wp.ndim == 0:
            return None
        else:
            return wp

    @check_model()
    def drifting(self):
        """


        :return:
        """
        return self.model.drifting

    @check_model()
    def ecea(self):
        """


        :return:
        """
        return self.model.ecea

    @check_model()
    def get_3D_drift(self, source=None, node=None, time_start=None, length=None):
        """




        :param source:
        :param node:
        :param time_start:
        :param length:
        Return the [X:][Y:][Z:] trail for a given node's drift from time_start backwards to
        a given length

        If no time given, assume the full time range
        """
        if source is None:
            source = self.model.drift_positions
        elif source == "intent":
            source = self.model.intent_positions
        else:
            raise (ValueError("Unknown 3D Drift Source"))
        time_start = time_start if time_start is not None else self.get_final_tmax()
        time_end = max(0 if length is None else (time_start - length), 0)

        if node is None:
            return source[:, :, time_start:time_end:-1].swapaxes(0, 1)
        else:
            return source[node, :, time_start:time_end:-1]
