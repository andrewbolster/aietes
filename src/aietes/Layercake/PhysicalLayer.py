# -*- coding: cp1252 -*-
###########################################################################
#
# Copyright (C) 2007 by Justin Eskesen and Josep Miquel Jornet Montana
# <jge@mit.edu> <jmjornet@mit.edu>
#
# Copyright: See COPYING file that comes with this distribution
#
# This file is part of AUVNetSim, a library for simulating acoustic
# networks of static and mobile underwater nodes, written in Python.
#
# AUVNetSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AUVNetSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AUVNetSim.  If not, see <http://www.gnu.org/licenses/>.
#
###########################################################################
from __future__ import division
import math
from copy import deepcopy

import numpy as np
import SimPy.Simulation as Sim
import scipy.special

from aietes.Tools import distance, DEBUG


# DEBUG = False


class PhysicalLayer(object):
    # Initialization of the layer

    def __init__(self, layercake, config, channel_event):
        self.config = config
        self.layercake = layercake
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)

        # Parameters initialization

        # Frequency Specifications
        self.freq = self.config["frequency"]  # In kHZ
        self.bandwidth = self.config["bandwidth"]  # In kHz
        self.band2bit = self.config["bandwidth_to_bit_ratio"]  # In bps/Hz
        self.variable_bandwidth = config["variable_bandwidth"]  # True or False

        # Power Specifications
        # In dB re uPa - The one when power control is not used
        self.transmit_power = self.config["transmit_power"]
        # In dB re uPa - To prevent unreal power values if power control does
        # not work properly
        self.max_output_power = self.config["transmit_power"]
        self.receive_power = self.config["receive_power"]  # In dBW
        self.listen_power = self.config["listen_power"]  # In dBW
        # True or False (if false = only one power level)
        self.variable_power = self.config["variable_power"]
        # Levels defined in terms of distance (m)
        self.level2distance = dict(self.config["var_power"])
        self.distance2level = {v: k for k, v in self.level2distance.items()}
        self.max_level = max(self.level2distance.keys())

        # Detection Specifications (dB)
        # Minimum signal to interference ratio to properly receive a packet
        self.SIR_threshold = self.config["threshold"]["SIR"]
        # Minimum signal to noise ratio to properly receive a packet
        self.SNR_threshold = self.config["threshold"]["SNR"]
        # Minimum signal to noise ratio to properly detect a packet
        self.LIS_threshold = self.config["threshold"]["LIS"]

        #
        self.receiving_threshold = db2linear(
            receiving_threshold(self.freq, self.bandwidth, self.SNR_threshold))
        self.listening_threshold = db2linear(
            listening_threshold(self.freq, self.bandwidth, self.LIS_threshold))

        # Definition of system parameters
        self.ambient_noise = db2linear(
            channel_noise_db(self.freq) + 10 * math.log10(self.bandwidth * 1e3))  # In linear scale (mw)
        # Initially, interference and noise are the same
        self.interference = self.ambient_noise

        # To emulate the half-duplex operation of the modem
        self.modem = Sim.Resource(name="a_modem")

        self.transducer = Transducer(self, self.ambient_noise, channel_event,
                                     self.layercake.get_real_current_position,
                                     self.SIR_threshold, self.on_successful_receipt)

        self.messages = []
        self.event = Sim.SimEvent("TransducerEvent" + self.layercake.hostname)

        # Values used to take statistics
        # In dB - Monitors the maximum transmission power used
        self.max_output_power_used = 0
        # Levels defined in terms of power (dB re uPa)
        self.distance2power = {}
        for level, distance in self.level2distance.iteritems():
            self.distance2power[level] = db2linear(
                distance2intensity(self.bandwidth, self.freq, distance, self.SNR_threshold))

        self.tx_energy = 0  # Energy consumed so far when tx (Joules)
        self.rx_energy = 0  # Energy consumed so far when rx (Joules)

        self.medium_speed = self.config['medium_speed']
        self.range = self.config['range']

    def dump_stats(self):
        """
        Throw up some useful information
        :return:
        """
        data = {
            "energy_tx": self.tx_energy,
            "energy_rx": self.rx_energy,
            "energy_tot": self.tx_energy + self.rx_energy,
            "collisions": len(self.transducer.collisions),

        }

        return data

    # Before transmitting, we should check if the system is idle or not
    def is_idle(self):
        if len(self.transducer.activeQ) > 0:
            if DEBUG:
                self.logger.debug(
                    "The channel is not idle. Currently receiving: " + str(
                        len(self.transducer.activeQ)) + " packet(s).")
            return False

        return True

    # Funtion called from the layers above. The MAC layer needs to transmit a
    # packet
    def transmit_packet(self, packet):

        if not self.is_idle():
            # The MAC protocol is the one that should check this before
            # transmitting
            self.logger.warn(
                "I should not do this... the channel is not idle!"
                "Trying to send {typ} to {dest}"
                "Currently have {q}".format(
                    typ=packet['type'],
                    dest=packet['dest'],
                    q=self.transducer.activeQ
                ))

        if self.variable_power:
            tx_range = self.level2distance[str(packet["level"])]
            power = distance2intensity(
                self.bandwidth, self.freq, tx_range, self.SNR_threshold)
        else:
            power = self.transmit_power

        if power > self.max_output_power_used:
            self.max_output_power_used = power

        if power > self.max_output_power:
            power = self.max_output_power

        new_transmission = OutgoingPacket(self)
        Sim.activate(
            new_transmission, new_transmission.transmit(packet, power))

    # Checks if there has been a collision
    def collision_detected(self):
        """


        :return:
        """
        return self.transducer.collision

    # When a packet has been received, we should pass it to the MAC layer
    def on_successful_receipt(self, packet):
        """

        :param packet:
        """
        self.layercake.mac.on_new_packet_received(packet)

    def level2delay(self, level):
        """

        :param level:
        :return:
        """
        distance = self.level2distance[str(level)]

        return distance / self.medium_speed

    def __str__(self):
        return "I'm " + self.layercake.hostname


class Transducer(Sim.Resource):

    """Transducer:  Uses a SimPy Resource model.
    --Incoming packets (from the channel): can request the resource
    and are allowed to interfere with one another.

    --Outgoing packets (coming from the modem): get queued by the modem until
    any outgoing transmission is completed.
    """

    def __init__(self, physical_layer, ambient_noise, channel_event, position_query, sir_thresh, on_success,
                 name="a_transducer"):
        # A resource with large capacity, because we don't want incoming messages to queue,
        # We want them to interfere.

        Sim.Resource.__init__(self, name=name, capacity=1000)

        self.physical_layer = physical_layer
        self.logger = physical_layer.logger.getChild(
            "{}".format(self.__class__.__name__))

        # Interference is initialized as ambient noise
        self.interference = ambient_noise

        # Setup our event listener
        self.channel_event = channel_event
        self.listener = AcousticEventListener(self)
        Sim.activate(self.listener, self.listener.listen(
            self.channel_event, position_query))

        # Function to call when we've received a packet
        self.on_success = on_success

        # SIR threshold
        self.SIR_thresh = sir_thresh

        # Controls the half-duplex behavior
        self.transmitting = False

        # Takes statistics about the collisions
        self.collision = False
        self.collisions = []

    # Override SimPy Resource's "_request" function to update SIR for all
    # incoming messages.
    def _request(self, arg):
        # We should update the activeQ
        Sim.Resource._request(self, arg)

        # "arg[1] is a reference to the IncomingPacket instance that has been just created
        new_packet = arg[1]

        # doom any newly incoming packets to failure if the transducer is
        # transmitting
        if self.transmitting:
            new_packet.doom()

        # Update all incoming packets' SIR
        self.interference += new_packet.power

        [i.update_interference(self.interference, new_packet.packet)
         for i in self.activeQ]

    # Override SimPy Resource's "_release" function to update SIR for all
    # incoming messages.
    def _release(self, arg):
        # "arg[1] is a reference to the IncomingPacket instance that just completed
        doomed = arg[1].doomed
        min_sir = arg[1].get_min_sir()
        new_packet = deepcopy(arg[1].packet)

        # Reduce the overall interference by this message's power
        self.interference -= arg[1].power
        # Prevent rounding errors
        if self.interference <= 0:
            self.interference = self.physical_layer.ambient_noise

        # Delete this from the transducer queue by calling the Parent form of
        # "_release"
        Sim.Resource._release(self, arg)

        # If it isn't doomed due to transmission & it is not interfered
        if min_sir > 0:
            if not doomed and linear2db(min_sir) >= self.SIR_thresh \
                    and arg[1].power >= self.physical_layer.receiving_threshold:

                ber = scipy.special.erfc(np.sqrt(linear2db(min_sir)))
                made_it = new_packet['length'] * ber

                # Properly received: enough power, not enough interference
                self.collision = False
                # self.logger.debug("received packet {}".format(new_packet))
                self.on_success(new_packet)

            elif arg[1].power >= self.physical_layer.receiving_threshold:
                # Too much interference but enough power to receive it: it
                # suffered a collision
                if self.physical_layer.layercake.hostname == new_packet["through"] \
                        or self.physical_layer.layercake.hostname == new_packet["dest"]:
                    self.collision = True
                    self.collisions.append(new_packet)
                    if DEBUG:
                        self.logger.debug(
                            "A " + new_packet["type"] + " packet to " + new_packet["through"] + " from " + new_packet[
                                'source']
                            + " was discarded due to interference.")
            else:
                # Not enough power to be properly received: just heard.
                if DEBUG and False:
                    if self.physical_layer.layercake.hostname in new_packet.items():
                        self.logger.info("Packet {id} from {src} to {dest} heard below reception threshold".format(
                            id=new_packet['ID'],
                            src=new_packet['source'],
                            dest=new_packet['through']
                        ))

        else:
            # This should never appear, and in fact, it doesn't, but just to
            # detect bugs (we cannot have a negative SIR in lineal scale).
            raise RuntimeError(
                "This really shouldn't happen: Negative min_sir from type {} from {} to {} through {} detected by {}".format(
                    new_packet["type"], new_packet["source"], new_packet["dest"], new_packet["through"],
                    self.physical_layer.layercake.hostname)
            )

    def on_transmit_begin(self):
        """


        """
        self.transmitting = True
        # doom all currently incoming packets to failure.
        [i.doom() for i in self.activeQ]

    def on_transmit_complete(self):
        """


        """
        self.transmitting = False


class IncomingPacket(Sim.Process):

    """IncomingPacket: A small class to represent a message being received
        by the transducer.  It keeps track of signal power as well as the
        maximum interference that occurs over its lifetime (worst case scenario).
        Powers are linear, not dB.
    """

    def __init__(self, power, packet, physical_layer):
        """

        :param power: float: linear power for transmission
        :param packet: object
        :param physical_layer: PhysicalLayer instance
        :return:
        """
        Sim.Process.__init__(self, name="({})RX from {}".format(
            physical_layer.layercake.hostname,
            packet['source']
        ))
        self.power = power  # linear
        self.packet = packet
        self.physical_layer = physical_layer
        self.doomed = False
        self.MaxInterference = 1

        # Need to add this info in for higher layers
        self.packet['rx_pwr_db'] = linear2db(self.power)

        if DEBUG:
            self.physical_layer.logger.debug("{type} {id} from {src} to {dest} recieved with power {pwr}".format(
                id=packet.get('ID'), src=packet['source'],
                dest=packet['dest'], pwr=power,
                type=packet['type']
            )
            )

    def update_interference(self, interference, packet):
        """

        :param interference:
        :param packet:
        """
        if interference > self.MaxInterference:
            self.MaxInterference = interference

    def receive(self, duration):
        """

        :param duration:
        """
        if self.power >= self.physical_layer.listening_threshold:
            # Otherwise I will not even notice that there are packets in the
            # network
            yield Sim.request, self, self.physical_layer.transducer
            yield Sim.hold, self, duration
            yield Sim.release, self, self.physical_layer.transducer

            # Even if a packet is not received properly, we have consumed power
            self.physical_layer.rx_energy += \
                db2linear(self.physical_layer.receive_power) * duration

    def get_min_sir(self):
        """
        Get the minimum SIR for successful reception
        :return: float
        """
        return self.power / (self.MaxInterference - self.power + 1)

    def doom(self):
        """


        """
        self.doomed = True


class OutgoingPacket(Sim.Process):

    """OutgoingPacket: A small class to represent a message being transmitted
        by the transducer.  It establishes the packet duration according to
        the bandwidth.

        Powers are in linear scale, not in dB, no they're bloody not....
    """

    def __init__(self, physical_layer):
        Sim.Process.__init__(self, name="({})TX".format(
            physical_layer.layercake.hostname
        ))
        self.physical_layer = physical_layer
        self.logger = physical_layer.logger.getChild(
            "{}".format(self.__class__.__name__))

    def transmit(self, packet, power):
        """

        :param packet:
        :param power:
        """
        yield Sim.request, self, self.physical_layer.modem

        # Create the acoustic event
        if self.physical_layer.variable_bandwidth:
            distance = self.physical_layer.level2distance[packet["level"]]
            bandwidth = distance2bandwidth(power, self.physical_layer.freq,
                                           distance, self.physical_layer.SNR_threshold)
        else:
            bandwidth = self.physical_layer.bandwidth

        # Real bit-rate
        bitrate = bandwidth * 1e3 * self.physical_layer.band2bit  # Bandwidth stored in KHz
        duration = packet["length"] / bitrate

        if DEBUG:
            self.logger.debug("{type} {id} to {dest} will take {s} to be transmitted".format(
                type=packet['type'],
                id=packet.get('ID'),  # SIL doesn't have ID
                s=duration,
                dest=packet['dest']
            ))

        # Need to add this info in for higher layers
        packet['tx_pwr_db'] = power

        self.physical_layer.transducer.channel_event.signal(
            {"pos": self.physical_layer.layercake.get_real_current_position,
             "power": power, "duration": duration, "frequency": self.physical_layer.freq,
             "packet": packet})

        # Hold onto the transducer for the duration of the transmission
        self.physical_layer.transducer.on_transmit_begin()

        yield Sim.hold, self, duration
        self.physical_layer.transducer.on_transmit_complete()

        # Release the modem when done
        yield Sim.release, self, self.physical_layer.modem

        power_w = db2linear(acoustic_power_db_per_upa(power))
        self.physical_layer.tx_energy += (power_w * duration)


#####################################################################
# Acoustic Event Listener
#####################################################################

class AcousticEventListener(Sim.Process):

    """AcousticEventListener: No physical analog.
    Waits for another node to send something and then activates
    an Arrival Scheduler instance.
    """

    def __init__(self, transducer):
        Sim.Process.__init__(self)
        self.transducer = transducer

    def listen(self, channel_event, position_query):
        """

        :param channel_event:
        :param position_query:
        """
        while True:
            yield Sim.waitevent, self, channel_event
            params = channel_event.signalparam
            sched = ArrivalScheduler(name="ArrivalScheduler" + self.name[-1])
            Sim.activate(sched, sched.schedule_arrival(
                self.transducer, params, position_query()))


#####################################################################
# Arrival Scheduler
#####################################################################

class ArrivalScheduler(Sim.Process):

    """ArrivalScheduler class: simulates the transit time of a message
    """

    def schedule_arrival(self, transducer, params, pos):
        """

        :param transducer:
        :param params:
        :param pos:
        """
        distance_to = distance(pos, params["pos"]())

        if distance_to > 0.01:  # I should not receive my own transmissions
            receive_power = params["power"] - \
                attenuation_db(params["frequency"], distance_to)
            # Speed of sound in water = 1482.0 m/s
            travel_time = distance_to / transducer.physical_layer.medium_speed

            packet = params['packet']
            if DEBUG:
                transducer.logger.debug("{type} from {source} to {dest} will take {t} to get to me {d}m away".format(
                    type=packet['type'], source=packet['source'],
                    dest=packet['dest'], t=travel_time, d=int(distance_to))
                )

            yield Sim.hold, self, travel_time

            new_incoming_packet = IncomingPacket(
                db2linear(receive_power), params["packet"], transducer.physical_layer)
            Sim.activate(
                new_incoming_packet, new_incoming_packet.receive(params["duration"]))


#####################################################################
# Propagation functions
#####################################################################

def attenuation_db(f, d):
    """

    Calculates the acoustic signal path loss
    as a function of frequency & distance

    .. math::
        A(f,d)=d^k * a(f)^d

        A_{dB}(f,d) = k * 10 \log(d) + d * 10\log(a(f))

    Cite Stojanovic2007

    >>> np.around(attenuation_db(10.0, 5000))
    -35.0

    f - Frequency in kHz
    d - Distance in m
    :param f: float
    :param d: float
    :returns: dB
    """

    k = 1.5  # "Practical Spreading" Stojanovic 08
    d_km = d / 1000.0

    a = thorpe(f)
    k_prod = k * linear2db(d_km)
    d_prod = d_km * a

    return k_prod + d_prod  # dB


def thorpe(f, heel=1):
    """
    Thorp's formula for attenuation_db rate (in dB/km) -> Changes depending on
    the frequency (kHz)
    From : "On the relationship between capacity and distance in an uncerwater acoustic communications channel: Stojanovic
    Using the heel creates a discontinuous graph. From the text:"
          [the first] formula is generally valid for frequencies abovce a few hundred Hz

    Cite Stojanovic2007

    >>> 51 < thorpe(200) < 51.5
    True
    >>> np.around(thorpe(200))
    51.0

    :param f: float kHz
    :param heel: float inversion f
    :return: float dB/km
    """

    f2 = f ** 2
    if f > heel:
        absorption_coeff = 0.11 * \
            (f2 / (1 + f2)) + 44.0 * (f2 / (4100 + f2)) + 0.000275 * f2 + 0.003
    else:
        absorption_coeff = 0.002 + 0.11 * (f2 / (1 + f2)) + 0.011 * f2
    return absorption_coeff  # dB/km


def channel_noise_db(f):
    """Noise(f)

    Calculates the ambient noise at current frequency

    Primarily Driven by Surface Wave activity. Approximation of (7) from Stojanovic2007



    :param f: float: Frequency in kHz
    :return: float: dB re uPA per Hz
    """

    # Noise approximation valid from a few kHz
    return 50 - (18 * np.log10(f))


def distance2bandwidth(i0, f, d, snr):
    """distance2bandwidth(P0, a, n, snr)

    Calculates the available bandwidth for the acoustic signal as
    a function of acoustic intensity, frequency and distance

    p
    f
    d
    snr

    :param i0:- transmit power in dB
    :param f:- Frequency in kHz
    :param d: - Distance to travel in m
    :param snr:- Signal to noise ratio in dB
    :return:
    """

    a = attenuation_db(f, d)
    n = channel_noise_db(f)

    return db2linear(i0 - snr - n - a - 30)  # In kHz


def distance2intensity(b, f, d, snr):
    """distance2Power(b, a, n, SNR)

    Calculates the acoustic intensity at the source as
    a function of bandwidth, frequency and distance

    :param b: float: Bandwidth in kHz
    :param f: float: Frequency in kHz
    :param d: float: Distance to travel in m
    :param snr: float: Signal to noise ratio in dB
    :return: float: acoustic intensity (dB@1uPa @1m?)
    """

    a = attenuation_db(f, d)
    n = channel_noise_db(f)
    b = linear2db(b * 1.0e3)

    return snr + a + n + b


def acoustic_power_db_per_upa(i):
    """acoustic_power_db_per_upa(P, dist)

    Calculates the acoustic power needed to create an acoustic intensity at a distance dist

    i - Created acoustic pressure in dB re uPa
    dist - Distance in m
    :param i:
    """
    # 170dB re uPa is the sound intensity created over a sphere of 1m by a
    # radiated acoustic power of 1 Watt with the source in the center
    i_ref = 172.0
    return i - i_ref


def listening_threshold(f, b, min_snr):
    """receiving_threshold(f, b)

    Calculates the minimum acoustic intensity that a node may be able to hear

    b - Bandwidth in kHz
    f - Frequency in kHz
    min_snr - Signal to noise ratio in dB
    :param f:
    :param b:
    :param min_snr:
    """

    n = channel_noise_db(f)
    b = linear2db(b * 1.0e3)

    return min_snr + n + b


def receiving_threshold(f, b, snr):
    """receiving_threshold(f, b)

    Calculates the minimum acoustic intensity that a packet should have to be properly received

    b - Bandwidth in kHz
    f - Frequency in kHz
    snr - Signal to noise ratio in dB
    :param f:
    :param b:
    :param snr:
    """

    n = channel_noise_db(f)
    b = linear2db(b * 1.0e3)

    return snr + n + b


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
