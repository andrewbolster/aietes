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
from collections import defaultdict

__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import cPickle
import errno
import functools
import logging
import math
import os
import pickle
import random
import re
import subprocess
import traceback
import validate
import warnings
from ast import literal_eval
from configobj import ConfigObj
from functools import partial
from inspect import getmembers, isfunction
from itertools import groupby
from operator import itemgetter
from shelve import DbfilenameShelf
from tempfile import mkdtemp
from time import time

import notify2
import numpy as np
from SimPy import SimulationStep as Sim
from colorlog import ColoredFormatter
from joblib import Memory

from aietes.Tools.humanize_time import seconds_to_str

np.seterr(**{'divide': 'ignore', 'invalid': 'ignore', 'over': 'ignore', 'under': 'ignore'})

memoize = Memory(cachedir=mkdtemp(), verbose=0)

DEBUG = False
FUDGED = False

_ROOT = os.path.abspath(os.path.dirname(__file__) + '/../')

_config_spec = '%s/configs/default.conf' % _ROOT
_results_dir = '%s/../../results/' % _ROOT
_config_dir = "%s/configs/" % _ROOT


_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0 * 1024.0,
          'KB': 1024.0, 'MB': 1024.0 * 1024.0}

var_rename_dict = {'CombinedBadMouthingPowerControl': 'MPC',
                   'CombinedSelfishTargetSelection': 'STS',
                   'CombinedTrust': 'Fair',
                   'Shadow': 'Shadow',
                   'SlowCoach': 'SlowCoach'}

metric_rename_dict = {
    'ADelay': "$Delay$",
    'ARXP': "$P_{RX}$",
    'ATXP': "$P_{TX}$",
    'RXThroughput': "$T^P_{RX}$",
    'TXThroughput': "$T^P_{TX}$",
    'PLR': '$PLR$',
    'INDD': '$INDD$',
    'INHD': '$INHD$',
    'Speed': '$Speed$'
}


def _vmb(vmkey):
    """Private.
    """
    global _proc_status, _scale
    # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
        # get vmkey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(vmkey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
        # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]] / (1024 * 1024)


def memory(since=0.0):
    """Return memory usage in Megabytes.
    :param since:
    """
    return _vmb('VmSize:') - since


def swapsize(since=0.0):
    """Return memory usage in Megabytes.
    :param since:
    """
    return _vmb('VmSwap:') - since


def resident(since=0.0):
    """Return resident memory usage in Megabytes.
    :param since:
    """
    return _vmb('VmRSS:') - since


def stacksize(since=0.0):
    """Return stack size in Megabytes.
    :param since:
    """
    return _vmb('VmStk:') - since


class SimTimeFilter(logging.Filter):
    """
    Brings Sim.now() into usefulness
    """

    def filter(self, record):
        record.simtime = Sim.now()
        return True


log_fmt = ColoredFormatter(
    "%(log_color)s%(simtime)-8.2f %(levelname)-6s %(name)s:%(message)s (%(lineno)d)",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
)
log_hdl = logging.StreamHandler()
log_hdl.setFormatter(log_fmt)
log_hdl.addFilter(SimTimeFilter())


def notify_desktop(message):
    """
    Thin Shim to notify when the simulations are complete, and not cry under no xsession
    :param message:
    :return:
    """
    try:
        if not notify2.is_initted():
            notify2.init("AIETES Simulation")
        notify2.Notification.new("AIETES Simulation", message, "dialog-information").show()
    except:
        pass


class ConfigError(Exception):
    """
    Raised when a configuration cannot be validated through ConfigObj/Validator
    Contains a 'status' with the boolean dict representation of the error
    """

    def __init__(self, message, errors=None):
        super(ConfigError, self).__init__(message)

        logging.critical("Invalid Config; Dying: %s" % message)
        self.status = message
        self.errors = errors

    def __str__(self):
        return repr(self.status)


#
# Magic Numbers
#
# 170dB re uPa is the sound intensity created over a sphere of 1m by a
# radiated acoustic power of 1 Watt with the source in the center
I_ref = 172.0
# Transducer Capacity (Arbitrary)
transducer_capacity = 1000
broadcast_address = 'Any'
LOGLEVELS = {'debug': logging.DEBUG,
             'info': logging.INFO,
             'warning': logging.WARNING,
             'error': logging.ERROR,
             'critical': logging.CRITICAL}

DEFAULT_CONVENTION = ['Alfa', 'Bravo', 'Charlie', 'Delta', 'Echo',
                      'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet',
                      'Kilo', 'Lima', 'Mike', 'November', 'Oscar',
                      'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango',
                      'Uniform', 'Victor', 'Whisky', 'X-ray', 'Yankee',
                      'Zulu']


#
# Measuring functions
#

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.1415926535897931
            :param v1:
            :param v2:
    """
    v1_u = unit(v1)
    v2_u = unit(v2)
    if np.allclose(v1_u, v2_u):
        return 0.0
    else:
        try:
            angle = np.arccos(np.dot(v1_u, v2_u))
        except FloatingPointError:
            logging.critical("FPE: 1:%f,2:%f", v1_u, v2_u)
            raise
    if np.isnan(angle):
        return np.pi
    else:
        return angle


def bearing(v):
    """radian angle between a given vector and 'north'
    :param v:
    """
    if np.all(v == 0):
        return 0.0
    else:
        return angle_between(v, np.array([1, 0]))


def distance(pos_a, pos_b, scale=1):
    """
    Return the distance between two positions
    :param pos_a:
    :param pos_b:
    :param scale:
    """
    try:
        return mag(pos_a - pos_b) * scale
    except (TypeError, FloatingPointError) as Err:
        logging.error("Type/FP Error on Distances (%s,%s): %s",
                      pos_a, pos_b, Err)
        raise


def mag(vector):
    """
    Return the magnitude of a given vector
    %timeit np.sqrt((uu1[0]-uu2[0])**2 +(uu1[1]-uu2[1])**2 +(uu1[2]-uu2[2])**2)-> 7.08us,
    %timeit np.linalg.norm(uu1-uu2) -> 11.7us.
    :param vector:
    """
    # FIXME  Might be faster unrolled
    return np.linalg.norm(vector)


def unit(vector):
    """
    Return the unit vector
    :param vector:
    """
    if mag(vector) == 0.0:
        return np.zeros_like(vector)
    else:
        return vector / mag(vector)


def agitate_position(position, maximum, var=10, minimum=None):
    """
    Fluff a position i by randn*var, limited to maximum/minimum
    :param position:
    :param maximum:
    :param var:
    :param minimum:
    :return:
    """
    if minimum is None:
        minimum = np.zeros(3) + var
    return np.max(
        np.vstack((
            minimum,
            np.min(
                np.vstack((
                    maximum - var,
                    position + np.random.randn(3) * var
                )),
                axis=0)
        )),
        axis=0
    )


def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def random_xy_vector():
    """
    Generates a random 2D vector in 3D space: (Planar random walk)
    this is a horrible cheat but it works.
    :return:
    """
    return random_three_vector()[0:2] + (0,)


def sixvec(xyz):
    """

    :param xyz:
    :return:
    """
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptsnew[3] = np.sqrt(xy + xyz[2] ** 2)
    # for elevation angle defined from Z-axis down
    ptsnew[4] = np.arctan2(np.sqrt(xy), xyz[2])
    ptsnew[5] = np.arctan2(xyz[1], xyz[0])
    return ptsnew


def spherical_distance(sixvec_a, sixvec_b):
    """

    :param sixvec_a:
    :param sixvec_b:
    :return:
    """
    return np.arccos(np.dot(sixvec_a, sixvec_b))


def add_ndarray_to_set(ndarray, input_list):
    """

    :param ndarray:
    :param input_list:
    :return:
    """
    in_list = False
    for element in input_list:
        if np.linalg.norm(ndarray - element) < 0.1:
            in_list = True

    if not in_list:
        input_list.append(ndarray)

    return input_list


#
# Lazy Testing functions
#


def records_check(pos_log):
    """

    :param pos_log:
    :return:
    """
    return [len([entry.time for entry in pos_log if entry.name == superentry.name]) for superentry in pos_log if
            superentry.time == 0]


def named_log(pos_log):
    """

    :param pos_log:
    :return:
    """
    return [object_log(pos_log, object_id) for object_id in node_ids(pos_log)]


def node_ids(pos_log):
    """

    :param pos_log:
    :return:
    """
    return (entry.object_id for entry in pos_log)


def object_log(pos_log, object_id):
    """

    :param pos_log:
    :param object_id:
    :return:
    """
    return [(entry.time, entry.position) for entry in pos_log if entry.object_id == object_id]


def db2linear(db):
    """

    :param db:
    :return:
    """
    return 10.0 ** (db / 10.0)


def linear2db(linear):
    """

    :param linear:
    :return:
    """
    return 10.0 * math.log10(linear + 0.0)


#
# Helper Classes
#

tree = lambda: defaultdict(tree)


class Dotdictify(dict):
    """

    :param value:
    :param kwargs:
    :raise TypeError:
    """
    marker = object()

    def __init__(self, value=None, **kwargs):
        super(Dotdictify, self).__init__(**kwargs)
        raise DeprecationWarning("Really shouldn't be using this any more!")
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError, "expected dict, got {}"

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Dotdictify):
            value = Dotdictify(value)
        try:
            dict.__setitem__(self, key, value)
        except TypeError:
            logging.error(
                "NOPE! {},{},{},{}".format(type(key), key, type(value), value))
            raise

    def __getitem__(self, key):
        found = self.get(key, Dotdictify.marker)
        if found is Dotdictify.marker:
            raise AttributeError("Key {} not found".format(key))

        return found

    def __copy__(self):
        newone = Dotdictify()
        newone.__dict__.update(dict(self.__dict__))
        return newone

    @staticmethod
    def __deepcopy__():
        raise (
            NotImplementedError, "Don't deep copy! This already deepcopy's on assignment")

    def __eq__(self, other):
        key_intersect = set(self.keys()).intersection(other.keys())

        # Basic sanity check
        if not len(key_intersect) > 0:
            return False
        if not key_intersect == self.keys() and key_intersect == other.keys():
            return False

        for k in key_intersect:
            try:
                if isinstance(self[k], np.ndarray) and not np.allclose(self[k], other[k]):
                    return False
                elif isinstance(self[k], Dotdictify) and not self[k] == other[k]:
                    return False
                else:
                    if not self[k] == other[k]:
                        return False
            except:
                print("Crashed on key {}{}:{}{}".format(
                    k, type(k), type(self[k]), type(other[k])))
                raise

        return True

    __setattr__ = __setitem__
    __getattr__ = __getitem__


class Dotdict(dict):
    """

    :param arg:
    :param kwargs:
    """

    def __init__(self, arg, **kwargs):
        super(Dotdict, self).__init__(**kwargs)
        for k in arg.keys():
            if type(arg[k]) is dict:
                self[k] = Dotdict(arg[k])
            else:
                self[k] = arg[k]

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return self.keys(), dir(dict(self))


class MemoryEntry(object):
    """

    :param object_id:
    :param position:
    :param velocity:
    :param distance:
    :param name:
    """

    def __init__(self, object_id, position, velocity, distance=None, name=None):
        self.object_id = object_id
        self.name = name
        self.position = position
        self.velocity = velocity
        self.distance = distance
        raise PendingDeprecationWarning("Pending Deletion")

    def __repr__(self):
        return "%s:%s:%s" % (self.name, self.position, self.distance)


class MapEntry(object):
    """

    :param object_id:
    :param position:
    :param velocity:
    :param name:
    :param distance:
    """

    def __init__(self, object_id, position, velocity, name=None, distance=None):
        self.object_id = object_id
        self.position = position
        self.distance = distance  # Not Always Used!
        self.velocity = velocity
        self.name = name
        self.time = Sim.now()

    def __repr__(self):
        return "%s:%s:%s" % (self.name, self.position, self.time)


class AutoSyncShelf(DbfilenameShelf):
    # default to newer pickle protocol and writeback=True

    """

    :param filename:
    :param protocol:
    :param writeback:
    """

    def __init__(self, filename, protocol=2, writeback=True):
        DbfilenameShelf.__init__(self, filename, protocol=protocol, writeback=writeback)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = str(key)
        DbfilenameShelf.__setitem__(self, key, value)
        self.sync()

    def __getitem__(self, key):
        if isinstance(key, int):
            key = str(key)
        return DbfilenameShelf.__getitem__(self, key)

    def __delitem__(self, key):
        if isinstance(key, int):
            key = str(key)
        DbfilenameShelf.__delitem__(self, key)
        self.sync()


def fudge_normal(value, stdev):
    # Override
    """

    :param value:
    :param stdev:
    :return: :raise ValueError:
    """
    if not FUDGED:
        return value

    # Deal with multiple inputs
    if hasattr(value, 'shape'):
        shape = value.shape
    elif isinstance(value, int) or isinstance(value, float):
        shape = 1
    elif isinstance(value, list):
        shape = len(value)
    else:
        raise ValueError("Cannot process value type %s:%s" %
                         (type(value), value))

    if stdev <= 0:
        return value
    else:
        return value + np.random.normal(0, stdev, shape)


def randomstr(length):
    """

    :param length:
    :return:
    """
    word = ''
    for i in range(length):
        word += random.choice(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    return word


def generate_names(count, naming_convention=None, existing_names=None):
    """

    :param count:
    :param naming_convention:
    :param existing_names:
    :return: :raise ConfigError:
    """
    if naming_convention is None:
        naming_convention = DEFAULT_CONVENTION

    if count > len(naming_convention):
        # If the naming convention can't provide unique names, bail
        raise ConfigError(
            "Not Enough Names in dictionary for number of nodes requested:%s/%s!" % (
                count, len(naming_convention))
        )

    node_names = []
    existing_names = existing_names if existing_names is not None else []

    for n in range(count):
        candidate_name = naming_convention[n]
        while candidate_name in node_names or candidate_name in existing_names:
            n += 1
            candidate_name = naming_convention[n]

        node_names.append(candidate_name)
    assert len(node_names) == count
    return node_names


def listfix(list_type, value):
    """

    :param list_type:
    :param value:
    :return:
    """
    if isinstance(value, list):
        return list_type(value[0])
    else:
        return list_type(value)


def range_grouper(data):
    """

    :param data:
    :return:
    """
    ranges = []
    data = filter(lambda (x): x is not None, data)
    for k, g in groupby(enumerate(data), lambda (i, x): i - x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    return ranges


def itersubclasses(cls, _seen=None):
    """
    itersubclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    :param cls:
    :param _seen:
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
        # get ALL (new-style) classes currently defined
        [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """

    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__()
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for ssub in itersubclasses(sub, _seen):
                yield ssub


def update_dict(d, keys, value, safe=False):
    """

    :param d:
    :param keys:
    :param value:
    :param safe:
    :raise KeyError:
    """
    for key in keys[:-1]:
        if key not in d and safe:
            raise KeyError("Attempting to update uninstantiated key")
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def list_functions(module):
    """

    :param module:
    :return:
    """
    return [o for o in getmembers(module) if isfunction(o[1])]


def unext(filename):
    """
    Tidier Basename
    :param filename:
    :return:
    """
    return os.path.splitext(os.path.basename(filename))[0]


def kwarger(**kwargs):
    """
    Lasy Dict Generation using Function Keywords
    :param kwargs:
    :return: dict
    """
    return kwargs


def uncpickle(filename):
    """
    UnCPickle, easy enough
    :param filename:
    :return: data
    """
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data


def mkcpickle(filename, thing):
    """
    CPickle, easy enough

    :param filename:
    :param thing:
    :return: stringbuffer: closed
    """
    with open(filename, 'wb') as f:
        data = cPickle.dump(thing, f)
    return f


def unpickle(filename):
    """
    UnPickle, easy enough
    :param filename:
    :return: data
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def mkpickle(filename, thing):
    """
    Pickle, easy enough
    :param filename:
    :param thing:
    :return: stringbuffer: closed
    """
    with open(filename, 'wb') as f:
        pickle.dump(thing, f)
    return f


def log_level_lookup(log_level):
    """

    :param log_level:
    :return:
    """
    if isinstance(log_level, str):
        return LOGLEVELS[log_level]
    else:
        # assume numeric/loglevel type, reverse lookup
        for k, v in LOGLEVELS.iteritems():
            if v == log_level:
                return k


def get_results_path(proposed_name, results_dir=None, make=False):
    """
    Default Path Joiner; provides a dir for results based on a proposed title and a
    base directory, i.e. generates /<results_dir>/<proposed_name>

    Can optionally make the directory as well.

    :param proposed_name:
    :param results_dir: Defaults to <repo_base>/results
    :param make: generate the directories on the FS
    :return: FQN Path
    :raises ValueError
    """
    # Have not been given a FQN Path: Assume to use the results directory
    if results_dir is None:
        results_dir = _results_dir

    if make:
        try:
            os.makedirs(results_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(results_dir):
                pass
            else:
                raise

    if proposed_name is None:
        raise ValueError("Proposed Name cannot be None")

    proposed_path = os.path.abspath(os.path.join(results_dir, proposed_name))

    return proposed_path


in_results = partial(os.path.join, _results_dir)


def get_config_file(config):
    """
    Return the full path to a config of a given filename if its in the default config path
    :param config:
    :return:
    """
    # If file is defined but not a file then something is wrong
    config = os.path.abspath(os.path.join(_config_dir, config))
    if not os.path.isfile(config):
        raise OSError(errno.ENOENT, "Given Source File {} is not present in {}".format(
            config, _config_dir
        ), config)

    return config


def get_config(source_config=None, config_spec=_config_spec):
    """
    Get a configuration, either using default values from aietes.configs or
        by taking a configobj compatible file path
    :param source_config:
    :param config_spec:
    """

    if source_config and not os.path.isfile(source_config):
        source_config_file = get_config_file(source_config)
    else:
        source_config_file = source_config
    config = ConfigObj(source_config_file,
                       configspec=config_spec,
                       stringify=True, interpolation=True)
    config_status = config.validate(validate.Validator(), copy=True)
    if not config_status:
        if source_config_file is None:
            raise RuntimeError("Configspec is Broken: %s" % config_spec)
        else:
            raise RuntimeError("Configspec doesn't match given input structure: %s"
                               % source_config_file)
    return config


def validate_config(config=None, final_check=False):
    """
    Generate valid confobj configuration information by interpolating a given config
    file with the defaults

    :param config:
    :param final_check:
    NOTE: This does not verify if any of the functionality requested in the config is THERE
    Only that the config 'makes sense' as requested.

    I.e. does not check if particular modular behaviour exists or not.
    """

    #
    # GENERIC CONFIG ACQUISITION
    #
    if not isinstance(config, ConfigObj):
        config = ConfigObj(
            config, configspec=_config_spec, stringify=True, interpolation=not final_check)
    else:
        raise ConfigError("Skipping configobj for final validation")
    config_status = config.validate(validate.Validator(), copy=not final_check)

    if not config_status:
        # If config_spec doesn't match the input, bail
        raise ConfigError(
            "Configspec doesn't match given input structure: %s" % config_status)

    return config


def try_x_times(x, exceptions_to_catch, exception_to_raise, fn):
    """

    :param x:
    :param exceptions_to_catch:
    :param exception_to_raise:
    :param fn:
    :return: :raise exception_to_raise:
    """

    @functools.wraps(fn)  # keeps name and docstring of old function
    def new_fn(*args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return: :raise exception_to_raise:
        """
        for i in xrange(x):
            try:
                return fn(*args, **kwargs)
            except exceptions_to_catch as e:
                print "Failed %d/%d: %s" % (i, x, e)
        raise exception_to_raise

    return new_fn


def try_forever(exceptions_to_catch, fn):
    """

    :param exceptions_to_catch:
    :param fn:
    :return:
    """

    @functools.wraps(fn)  # keeps name and docstring of old function
    def new_fn(*args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        count = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except exceptions_to_catch as e:
                count += 1
                print "Failed %d: %s" % (count, e)

    return new_fn

def npuniq(a):
    """
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    :param a:
    :return:
    """
    return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])

def timeit():
    """
    Simple Function Timer Decorator that still returns the results

    :return: decorator
    """

    def decorator(func):
        """

        :param func:
        :return:
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """

            :param args:
            :param kwargs:
            :return:
            """
            start = time()
            res = func(*args, **kwargs)
            print("%s (%s)" % (func.__name__, time() - start))
            return res

        return wrapper

    return decorator


def are_equal_waypoints(wps):
    """Compare Waypoint Objects as used by WaypointMixin ([pos],prox)
        Will exclude 'None' records in wps and only compare valid waypoint lists
    :param wps:
    """
    poss = [[w.position for w in wp] for wp in wps if wp is not None]
    proxs = [[w.prox for w in wp] for wp in wps if wp is not None]
    for pos in poss:
        if not np.array_equal(pos, poss[0]):
            return False
    for prox in proxs:
        if not np.array_equal(prox, proxs[0]):
            return False

    return True


def get_latest_aietes_datafile(base_dir=None):
    """

    :param base_dir:
    :return: :raise ValueError:
    """
    fqp = os.getcwd() if base_dir is None else base_dir
    candidate_data_files = os.listdir(fqp)
    candidate_data_files = [
        f for f in candidate_data_files
        if is_valid_aietes_datafile(f)
    ]
    candidate_data_files.sort(key=os.path.getmtime, reverse=True)
    if len(candidate_data_files) == 0:
        raise ValueError(
            "There are no valid datafiles in the working directory:%s" % os.getcwd())
    return os.path.join(fqp, candidate_data_files[0])


def is_valid_aietes_datafile(filename):
    # TODO This isn't a very good test...
    """

    :param filename:
    :return:
    """
    test = re.compile(".npz$")
    return test.search(filename)


def literal_eval_walk(node, tabs=0):
    """

    :param node:
    :param tabs:
    """
    for key, item in node.items():
        if isinstance(item, dict):
            try:
                literal_eval(str(item))
            except:
                print '*' * tabs, key
                tabs += 1
                literal_eval_walk(item, tabs)
                tabs -= 1
        else:
            try:
                literal_eval(str(item))

            except:
                print '*' * tabs, key, "EOL FAIL"


def map_paths(paths):
    subdirs = reduce(list.__add__, [filter(os.path.isdir,
                                           map(lambda p: os.path.join(path, p),
                                               os.listdir(path)
                                               )
                                           ) for path in paths])
    return subdirs


def map_levels(df, dct, level=0):
    """
    Renames items in a index/multiindex of a DataFrame in place
    :param df: pd.DataFrame
    :param dct: dict: Pairs of Item Names to Swap
    :param level: int: pd.MultiIndex level
    :return:
    """
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i == level else names
                      for i, names in enumerate(index.levels)], inplace=True)


def categorise_dataframe(df):
    # Categories work better as indexes
    for obj_key in df.keys()[df.dtypes == object]:
        try:
            df[obj_key] = df[obj_key].astype('category')
        except TypeError:
            print("Couldn't categorise {}".format(obj_key))
            pass
    return df


def non_zero_rows(df):
    return df[~(df == 0).all(axis=1)]


def try_to_open(filename):
    try:
        if sys.platform == 'linux2':
            subprocess.call(["xdg-open", filename])
        else:
            os.startfile(filename)
    except Exception:
        warnings.warn(traceback.format_exc())


from cStringIO import StringIO
import sys


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
