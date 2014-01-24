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

import math
import functools
import os
import random
import logging
import re
from inspect import getmembers, isfunction
from itertools import groupby
from operator import itemgetter
from pprint import pformat
import numpy as np
from datetime import datetime as dt
from configobj import ConfigObj
import validate

from SimPy import SimulationStep as Sim

np.seterr(all='raise')
# from os import urandom as randomstr #Provides unicode random String


debug = False
FUDGED = True

_ROOT = os.path.abspath(os.path.dirname(__file__) + '/../')

_config_spec = '%s/configs/default.conf' % _ROOT
_results_dir = '%s/../../results/' % _ROOT


class ConfigError(Exception):
    """
    Raised when a configuration cannot be validated through ConfigObj/Validator
    Contains a 'status' with the boolean dict representation of the error
    """

    def __init__(self, value):
        logging.critical("Invalid Config; Dying: %s" % value)
        self.status = value

    def __str__(self):
        return repr(self.status)

#
# Magic Numbers
#
# 170dB re uPa is the sound intensity created over a sphere of 1m by a
# radiated acoustic power of 1 Watt with the source in the center
I_ref = 172.0
# Speed of sound in water (m/s)
speed_of_sound = 1482
# Transducer Capacity (Arbitrary)
transducer_capacity = 1000
broadcast_address = 'Any'
LOGLEVELS = {'debug': logging.DEBUG,
             'info': logging.INFO,
             'warning': logging.WARNING,
             'error': logging.ERROR,
             'critical': logging.CRITICAL}

DEFAULT_CONVENTION = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon',
                      'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
                      'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron',
                      'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon',
                      'Phi', 'Chi', 'Psi', 'Omega']
DEFAULT_CONVENTION = ['Alfa', 'Bravo', 'Charlie', 'Delta', 'Echo',
                      'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet',
                      'Kilo', 'Lima', 'Mike', 'November', 'Oscar',
                      'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango',
                      'Uniform', 'Victor', 'Whisky', 'X-ray', 'Yankee',
                      'Zulu']


#
# Measuring functions
#

def distance(pos_a, pos_b, scale=1):
    """
    Return the distance between two positions
    """
    try:
        return mag(pos_a - pos_b) * scale
    except TypeError as Err:
        logging.error("TypeError on Distances (%s,%s): %s" % (pos_a, pos_b, Err))
        raise


def mag(vector):
    """
    Return the magnitude of a given vector
    """
    return np.linalg.norm(vector)


def unit(vector):
    """
    Return the unit vector
    """
    if mag(vector) == 0.0:
        return np.zeros_like(vector)
    else:
        return vector / mag(vector)


def sixvec(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptsnew[3] = np.sqrt(xy + xyz[2] ** 2)
    ptsnew[4] = np.arctan2(np.sqrt(xy), xyz[2])  # for elevation angle defined from Z-axis down
    ptsnew[5] = np.arctan2(xyz[1], xyz[0])
    return ptsnew


def spherical_distance(sixvec_a, sixvec_b):
    return np.arccos(np.dot(sixvec_a, sixvec_b))


def add_ndarray_to_set(ndarray, list):
    in_list = False
    for element in list:
        if np.linalg.norm(ndarray - element) < 0.1:
            in_list = True

    if not in_list:
        list.append(ndarray)

    return list


#
# Lazy Testing functions
#


def recordsCheck(pos_log):
    return [len([entry.time for entry in pos_log if entry.name == superentry.name]) for superentry in pos_log if
            superentry.time == 0]


def namedLog(pos_log):
    return [objectLog(pos_log, object_id) for object_id in nodeIDs(pos_log)]


def nodeIDs(pos_log):
    return (entry.object_id for entry in pos_log)


def objectLog(pos_log, object_id):
    return [(entry.time, entry.position) for entry in pos_log if entry.object_id == object_id]

#
# Propagation functions
#


def Attenuation(f, d):
    """Attenuation(P0,f,d)

    Calculates the acoustic signal path loss
    as a function of frequency & distance

    f - Frequency in kHz
    d - Distance in m
    """

    f2 = f ** 2
    k = 1.5  # Practical Spreading, see http://rpsea.org/forums/auto_stojanovic.pdf
    DistanceInKm = d / 1000.0

    # Thorp's formula for attenuation rate (in dB/km) -> Changes depending on the frequency
    if f > 1:
        absorption_coeff = 0.11 * (f2 / (1 + f2)) + 44.0 * (f2 / (4100 + f2)) + 0.000275 * f2 + 0.003
    else:
        absorption_coeff = 0.002 + 0.11 * (f2 / (1 + f2)) + 0.011 * f2

    return k * Linear2DB(d) + DistanceInKm * absorption_coeff


def Noise(f):
    """Noise(f)

    Calculates the ambient noise at current frequency

    f - Frequency in kHz
    """

    # Noise approximation valid from a few kHz
    return 50 - 18 * math.log10(f)


def distance2Bandwidth(I0, f, d, SNR):
    """distance2Bandwidth(P0, A, N, SNR)

    Calculates the available bandwidth for the acoustic signal as
    a function of acoustic intensity, frequency and distance

    I0 - Transmit power in dB
    f - Frequency in kHz
    d - Distance to travel in m
    SNR - Signal to noise ratio in dB
    """

    A = Attenuation(f, d)
    N = Noise(f)

    return DB2Linear(I0 - SNR - N - A - 30)  # In kHz


def distance2Intensity(B, f, d, SNR):
    """distance2Power(B, A, N, SNR)

    Calculates the acoustic intensity at the source as
    a function of bandwidth, frequency and distance

    B - Bandwidth in kHz
    f - Frequency in kHz
    d - Distance to travel in m
    SNR - Signal to noise ratio in dB
    """

    A = Attenuation(f, d)
    N = Noise(f)
    B = Linear2DB(B * 1.0e3)

    return SNR + A + N + B


def AcousticPower(I):
    """AcousticPower(P, dist)

    Calculates the acoustic power needed to create an acoustic intensity at a distance dist

    I - Created acoustic pressure
    dist - Distance in m
    """
    return I - I_ref


def ListeningThreshold(f, B, minSNR):
    """ReceivingThreshold(f, B)

    Calculates the minimum acoustic intensity that a node may be able to hear

    B - Bandwidth in kHz
    f - Frequency in kHz
    minSNR - Signal to noise ratio in dB
    """

    N = Noise(f)
    B = Linear2DB(B * 1.0e3)

    return minSNR + N + B


def ReceivingThreshold(f, B, SNR):
    """ReceivingThreshold(f, B)

    Calculates the minimum acoustic intensity that a packet should have to be properly received

    B - Bandwidth in kHz
    f - Frequency in kHz
    SNR - Signal to noise ratio in dB
    """

    N = Noise(f)
    B = Linear2DB(B * 1.0e3)

    return SNR + N + B


def DB2Linear(dB):
    return 10.0 ** (dB / 10.0)


def Linear2DB(Linear):
    return 10.0 * math.log10(Linear + 0.0)

#
# Helper Classes
#


class dotdictify(dict):
    marker = object()

    def __init__(self, value=None, **kwargs):
        super(dotdictify, self).__init__(**kwargs)
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError, 'expected dict'

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, dotdictify):
            value = dotdictify(value)
        dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        found = self.get(key, dotdictify.marker)
        if found is dotdictify.marker:
            found = dotdictify()
            dict.__setitem__(self, key, found)
        return found

    __setattr__ = __setitem__
    __getattr__ = __getitem__


class dotdict(dict):
    def __init__(self, arg, **kwargs):
        super(dotdict, self).__init__(**kwargs)
        for k in arg.keys():
            if type(arg[k]) is dict:
                self[k] = dotdict(arg[k])
            else:
                self[k] = arg[k]

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return self.keys(), dir(dict(self))


class memory_entry():
    def __init__(self, object_id, position, velocity, distance=None, name=None):
        self.object_id = object_id
        self.name = name
        self.position = position
        self.velocity = velocity
        self.distance = distance

    def __repr__(self):
        return "%s:%s:%s" % (self.name, self.position, self.distance)


class map_entry():
    def __init__(self, object_id, position, velocity, name=None, distance=None):
        self.object_id = object_id
        self.position = position
        self.distance = distance  # Not Always Used!
        self.velocity = velocity
        self.name = name
        self.time = Sim.now()

    def __repr__(self):
        return "%s:%s:%s" % (self.object_id, self.position, self.time)


def fudge_normal(value, stdev):
    # Override
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
        raise ValueError("Cannot process value type %s:%s" % (type(value), value))

    if stdev <= 0:
        return value
    else:
        return value + np.random.normal(0, stdev, shape)


def randomstr(length):
    word = ''
    for i in range(length):
        word += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    return word


def nameGeneration(count, naming_convention=None, existing_names=None):
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
        candidate_name = naming_convention[np.random.randint(0, len(naming_convention))]

        while candidate_name in node_names or candidate_name in existing_names:
            candidate_name = naming_convention[np.random.randint(0, len(naming_convention))]

        node_names.append(candidate_name)
    assert len(node_names) == count
    return node_names


def listfix(list_type, value):
    if isinstance(value, list):
        return list_type(value[0])
    else:
        return list_type(value)


def timestamp():
    return dt.now().strftime('%Y-%m-%d-%H-%M-%S.aietes')


def grouper(data):
    ranges = []
    for key, group in groupby(enumerate(data), lambda (index, item): index - item):
        group = map(itemgetter(1), group)
        if len(group) > 1:
            ranges.append(range(group[0], group[-1]))
        else:
            ranges.append(group[0])
    return ranges


def range_grouper(data):
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
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub


def updateDict(d, keys, value, safe=False):
    for key in keys[:-1]:
        if not d.has_key(key) and safe:
            raise KeyError("Attempting to update uninstantiated key")
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def list_functions(module):
    return [o for o in getmembers(module) if isfunction(o[1])]


def unext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def kwarger(**kwargs):
    return kwargs


def log_level_lookup(log_level):
    if isinstance(log_level, str):
        return LOGLEVELS[log_level]
    else:
        #assume numeric/loglevel type, reverse lookup
        for k, v in LOGLEVELS.iteritems():
            if v == log_level:
                return k


def validateConfig(config=None, final_check=False):
    """
    Generate valid configuration information by interpolating a given config
    file with the defaults

    NOTE: This does not verify if any of the functionality requested in the config is THERE
    Only that the config 'makes sense' as requested.

    I.e. does not check if particular modular behaviour exists or not.
    """

    #
    # GENERIC CONFIG ACQUISITION
    #
    if not isinstance(config, ConfigObj):
        config = ConfigObj(config, configspec=_config_spec, stringify=True, interpolation=not final_check)
    else:
        raise ConfigError("Skipping configobj for final validation")
    config_status = config.validate(validate.Validator(), copy=not final_check)

    if not config_status:
        # If config_spec doesn't match the input, bail
        raise ConfigError("Configspec doesn't match given input structure: %s" % config_status)

    return config


def try_x_times(x, exceptions_to_catch, exception_to_raise, fn):
    @functools.wraps(fn) #keeps name and docstring of old function
    def new_fn(*args, **kwargs):
        for i in xrange(x):
            try:
                return fn(*args, **kwargs)
            except exceptions_to_catch as e:
                print "Failed %d/%d: %s" % (i, x, e)
        raise exception_to_raise

    return new_fn


def try_forever(exceptions_to_catch, fn):
    @functools.wraps(fn) #keeps name and docstring of old function
    def new_fn(*args, **kwargs):
        count = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except exceptions_to_catch as e:
                count += 1
                print "Failed %d: %s" % (count, e)

    return new_fn


def are_equal_waypoints(wps):
    """Compare Waypoint Objects as used by WaypointMixin ([pos],prox)
        Will exclude 'None' records in wps and only compare valid waypoint lists
    """
    retval = True
    poss = [[w[0] for w in wp] for wp in wps if wp is not None]
    proxs = [[w[1] for w in wp] for wp in wps if wp is not None]
    for pos in poss:
        if not np.array_equal(pos, poss[0]):
            retval = False
    for prox in proxs:
        if not np.array_equal(prox, proxs[0]):
            retval = False

    if retval is False:
        logging.error(pformat(zip(poss, proxs)))
    return retval


def get_latest_aietes_datafile(dir=None):
    fqp = os.getcwd() if dir is None else dir
    candidate_data_files = os.listdir(fqp)
    candidate_data_files = [f for f in candidate_data_files if is_valid_aietes_datafile(f)]
    candidate_data_files.sort(key=os.path.getmtime, reverse=True)
    if len(candidate_data_files) == 0:
        raise ValueError("There are no valid datafiles in the working directory:%s" % os.getcwd())
    return os.path.join(fqp, candidate_data_files[0])


def is_valid_aietes_datafile(file):
    #TODO This isn't a very good test...
    test = re.compile(".npz$")
    return test.search(file)


def angle_between(v1, v2, ndim=3):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit(v1[0:ndim - 1])
    v2_u = unit(v2[0:ndim - 1])
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle