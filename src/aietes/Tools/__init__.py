from SimPy import SimulationStep as Sim
import math
import random
import logging
import numpy as np
np.seterr(over='raise')
#from os import urandom as randomstr #Provides unicode random String

logging.basicConfig(level=logging.DEBUG)
baselogger = logging.getLogger('SIM')

debug = False

class ConfigError(Exception):
    """
    Raised when a configuration cannot be validated through ConfigObj/Validator
    Contains a 'status' with the boolean dict representation of the error
    """
    def __init__(self,value):
        baselogger.critical("Invalid Config; Dying: %s" %value)
        self.status=value
    def __str__(self):
        return repr(self.status)


#####################################################################
# Magic Numbers
#####################################################################
# 170dB re uPa is the sound intensity created over a sphere of 1m by a radiated acoustic power of 1 Watt with the source in the center
I_ref = 172.0
# Speed of sound in water (m/s)
speed_of_sound= 1482
#Transducer Capacity (Arbitrary)
transducer_capacity= 1000
broadcast_address='Any'
LOGLEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

DEFAULT_CONVENTION = ['Alpha','Beta','Gamma','Delta','Epsilon',
                      'Zeta','Eta','Theta','Iota','Kappa',
                      'Lambda','Mu','Nu','Xi','Omicron',
                      'Pi','Rho','Sigma','Tau','Upsilon',
                      'Phi','Chi','Psi','Omega']
DEFAULT_CONVENTION = ['Alfa','Bravo','Charlie','Delta','Echo',
                      'Foxtrot','Golf','Hotel','India','Juliet',
                      'Kilo','Lima','Mike','November','Oscar',
                      'Papa','Quebec','Romeo','Sierra','Tango',
                      'Uniform','Victor','Whisky','X-ray','Yankee',
                      'Zulu']


#####################################################################
# Measuring functions
#####################################################################

def distance(pos_a, pos_b, scale=1):
    """
    Return the distance between two positions
    """
    try:
        return mag(pos_a - pos_b) * scale
    except TypeError as Err:
        logging.error("TypeError on Distances (%s,%s): %s"%(pos_a, pos_b, Err))
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
	if mag(vector)==0.0:
		return np.zeros_like(vector)
	else:
		return vector / np.linalg.norm(vector)

#####################################################################
# Lazy Testing functions
#####################################################################

def recordsCheck(pos_log):
    return [ len([ entry.time for entry in pos_log if entry.name == superentry.name]) for superentry in pos_log if superentry.time == 0]

def namedLog(pos_log):
    return [ objectLog(pos_log,object_id) for object_id in nodeIDs(pos_log)]

def nodeIDs(pos_log):
    return { entry.object_id for entry in pos_log }

def objectLog(pos_log,object_id):
    return [(entry.time,entry.position) for entry in pos_log if entry.object_id == object_id]

#####################################################################
# Propagation functions
#####################################################################

def Attenuation(f, d):
    """Attenuation(P0,f,d)

    Calculates the acoustic signal path loss
    as a function of frequency & distance

    f - Frequency in kHz
    d - Distance in m
    """

    f2 = f**2
    k = 1.5 # Practical Spreading, see http://rpsea.org/forums/auto_stojanovic.pdf
    DistanceInKm = d/1000.0

    # Thorp's formula for attenuation rate (in dB/km) -> Changes depending on the frequency
    if f > 1:
        absorption_coeff = 0.11*(f2/(1+f2)) + 44.0*(f2/(4100+f2)) + 0.000275*f2 + 0.003
    else:
        absorption_coeff = 0.002+0.11*(f2/(1+f2)) + 0.011*f2

    return k*Linear2DB(d) + DistanceInKm*absorption_coeff


def Noise(f):
    """Noise(f)

    Calculates the ambient noise at current frequency

    f - Frequency in kHz
    """

    # Noise approximation valid from a few kHz
    return 50 - 18*math.log10(f)


def distance2Bandwidth(I0, f, d, SNR):
    """distance2Bandwidth(P0, A, N, SNR)

    Calculates the available bandwidth for the acoustic signal as
    a function of acoustic intensity, frequency and distance

    I0 - Transmit power in dB
    f - Frequency in kHz
    d - Distance to travel in m
    SNR - Signal to noise ratio in dB
    """

    A = Attenuation(f,d)
    N = Noise(f)

    return DB2Linear(I0-SNR-N-A-30) #In kHz


def distance2Intensity(B, f, d, SNR):
    """distance2Power(B, A, N, SNR)

    Calculates the acoustic intensity at the source as
    a function of bandwidth, frequency and distance

    B - Bandwidth in kHz
    f - Frequency in kHz
    d - Distance to travel in m
    SNR - Signal to noise ratio in dB
    """

    A = Attenuation(f,d)
    N = Noise(f)
    B = Linear2DB(B*1.0e3)

    return SNR + A + N + B


def AcousticPower(I):
    """AcousticPower(P, dist)

    Calculates the acoustic power needed to create an acoustic intensity at a distance dist

    I - Created acoustic pressure
    dist - Distance in m
    """
    return I-I_ref


def ListeningThreshold(f, B, minSNR):
    """ReceivingThreshold(f, B)

    Calculates the minimum acoustic intensity that a node may be able to hear

    B - Bandwidth in kHz
    f - Frequency in kHz
    minSNR - Signal to noise ratio in dB
    """

    N = Noise(f)
    B = Linear2DB(B*1.0e3)

    return minSNR + N + B


def ReceivingThreshold(f, B, SNR):
    """ReceivingThreshold(f, B)

    Calculates the minimum acoustic intensity that a packet should have to be properly received

    B - Bandwidth in kHz
    f - Frequency in kHz
    SNR - Signal to noise ratio in dB
    """

    N = Noise(f)
    B = Linear2DB(B*1.0e3)

    return SNR + N + B


def DB2Linear(dB):
    return 10.0**(dB/10.0)


def Linear2DB(Linear):
    return 10.0*math.log10(Linear+0.0)

#####################################################################
# Helper Classes
#####################################################################
class dotdictify(dict):
    marker = object()
    def __init__(self, value=None):
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
    def __init__(self,arg):
        for k in arg.keys():
            if (type(arg[k]) is dict):
                self[k]=dotdict(arg[k])
            else:
                self[k]=arg[k]
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __dir__(self):
        return self.keys(),dir(dict(self))

class memory_entry():
    def __init__(self,object_id,position,distance,name=None):
        self.object_id=object_id
        self.name=name
        self.position=position
        self.distance=distance
    def __repr__(self):
        return "%s:%s:%s"%(self.name,self.position,self.distance)

class map_entry():
    def __init__(self,object_id,position,name=None):
        self.object_id=object_id
        self.position=position
        self.name=name
        self.time=Sim.now()
    def __repr__(self):
        return "%s:%s:%s"%(self.object_id,self.position,self.time)

def fudge_normal(value,stdev):
    #Deal with multiple inputs
    if hasattr(value,'shape'):
        shape = value.shape
    elif isinstance(value, int) or isinstance(value,float):
        shape = 1
    elif isinstance(value, list):
        shape = len(value)
    else:
        raise ValueError("Cannot process value type %s:%s"%(type(value),value))
    if stdev <=0:
        return value
    else:
        return value + np.random.normal(0,stdev,shape)

def randomstr(length):

        word = ''
        for i in range(length):
            word += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        return word

def nameGeneration(count, naming_convention = None):

    if naming_convention is None:
        naming_convention = DEFAULT_CONVENTION

    if count > len(naming_convention):
        # If the naming convention can't provide unique names, bail
        raise ConfigError(
            "Not Enough Names in dictionary for number of nodes requested:%s/%s!"%(
            count,len(naming_convention))
        )

    node_names = []

    for n in range(count):
        candidate_name = naming_convention[np.random.randint(0,len(naming_convention))]

        while candidate_name in node_names:
            candidate_name= naming_convention[np.random.randint(0,len(naming_convention))]

        node_names.append(candidate_name)

    return node_names

def listfix(type,value):
	if isinstance(value,list):
		return type(value[0])
	else:
		return value


