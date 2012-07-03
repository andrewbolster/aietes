import math
import logging

logging.basicConfig(level=logging.DEBUG)
baselogger = logging.getLogger('AIETES')

#####################################################################
# Magic Numbers
#####################################################################
# 170dB re uPa is the sound intensity created over a sphere of 1m by a radiated acoustic power of 1 Watt with the source in the center
I_ref = 172.0
# Speed of sound in water (m/s)
speed_of_sound= 1482
#Transducer Capacity (Arbitrary)
transducer_capacity= 1000

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
    DistanceInKm = d/1000

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
