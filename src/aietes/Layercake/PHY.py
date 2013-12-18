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

from aietes.Tools import *
from Packet import PHYPacket

debug = False

#####################################################################
# Physical Layer
#####################################################################
class PHY():
    """A Generic Class for Physical interface layers
    """

    def __init__(self, layercake, channel_event, config):
        """Initialise with defaults from PHYconfig
        :Frequency Specifications
            frequency (kHz)
            bandwidth (kHz)
            bandwidth_to_bit_ratio (bps/Hz)
            variable_bandwidth (bool)
        :Power Specifications
            transmit_power (dB re 1 uPa @1m)
            max_transmit_power (dB re 1 uPa @1m)
            receive_power (dBw)
            listen_power (dBw)
            var_power (None|,
                { 'levelToDistance':
                    {'(n)':'(km)',(*)},
                }
            )
        :Initial Detection Specifications
            threshold (
                { 'SIR': (dB),
                  'SNR': (dB),# sufficient to receive
                  'LIS': (dB) # sufficient to detect
                }
            }
        """
        ##############################
        #Generic Spec
        ##############################
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)
        self.logger.info('creating instance:%s' % config)
        #Inherit values from config
        for k, v in config.items():
            self.logger.info("Updating: [%s]=%s" % (k, v))
            setattr(self, k, v)
        self.layercake = layercake

        ##############################
        #Inferred System Parameters
        ##############################
        self.threshold['receive'] = DB2Linear(ReceivingThreshold(self.frequency, self.bandwidth, self.threshold['SNR']))
        self.threshold['listen'] = DB2Linear(ListeningThreshold(self.frequency, self.bandwidth, self.threshold['LIS']))
        self.ambient_noise = DB2Linear(Noise(self.frequency) + 10 * math.log10(self.bandwidth * 1e3)) #In linear scale
        self.interference = self.ambient_noise
        self.collision = False

        ##############################
        #Generate modem/transd etc
        ##############################
        self.modem = Sim.Resource(name=self.__class__.__name__)
        self.transducer = Transducer(self)
        self.messages = []

        ##############################
        #Statistics
        ##############################
        self.max_output_power_used = 0
        self.tx_energy = 0
        self.rx_energy = 0

    def isIdle(self):
        """Before TX, check if the transducer activeQ (inherited from Sim.Resource) is empty i.e
        Are we recieving?
        """
        if len(self.transducer.activeQ) > 0:
            self.logger.info(
                "The Channel is not idle: %d packets currently in flight" % str(len(self.transducer.activeQ)))
            return False
        return True

    def send(self, FromAbove):
        """Function called from upper layers to send packet
        """
        if not self.isIdle():
            self.PrintMessage(
                "I should not do this... the channel is not idle!") # The MAC protocol is the one that should check this before transmitting

        self.collision = False

        if hasattr(self, "variable_power") and self.variable_power:
            tx_range = self.level2distance[FromAbove.level]
            power = distance2Intensity(self.bandwidth, self.frequency, tx_range, self.SNR_threshold)
        else:
            power = self.transmit_power

        if power > self.max_output_power_used:
            self.max_output_power_used = power

        #generate PHYPacket with set power
        packet = PHYPacket(packet=FromAbove, power=power)
        if debug: self.logger.debug("PHY Packet, %s, sent to %s, with power %s" % (packet.data, packet.next_hop, power))
        Sim.activate(packet, packet.send(phy=self))

    def bandwidth_to_bit(self, bandwidth):
        return bandwidth * 1e3 * self.bandwidth_to_bit_ratio

#####################################################################
# Transducer
#####################################################################
class Transducer(Sim.Resource):
    """Packets request the resource and can interfere
    """

    def __init__(self, phy,
                 name="Transducer"):

        Sim.Resource.__init__(self, name=name, capacity=transducer_capacity)
        self.logger = phy.logger.getChild("%s" % self.__class__.__name__)
        self.logger.info('creating instance')

        self.phy = phy
        self.layercake = self.phy.layercake
        self.host = self.layercake.host

        self.transmitting = False
        self.collisions = []
        self.channel_event = self.layercake.channel_event
        self.interference = self.phy.interference

        self.threshold = self.phy.threshold

        ##############################
        #Configure event listener
        ##############################
        self.listener = AcousticEventListener(self)
        Sim.activate(
            self.listener,
            self.listener.listen(
                self.channel_event,
                self.host.getPos)
        )

    def updateInterference(self, packet):
        """ Update interferences of the active queue
        """
        if self.transmitting:
            packet.Doom()

        self.interference += packet.power

        [x.updateInterference(self.interference) for x in self.activeQ]

    def _request(self, arg):
        """Overiding SimPy's to update interference information upon queuing of a new incoming packet from the channel
        """
        Sim.Resource._request(self, arg)
        #Arg[1] is a reference to the newly queued incoming packet
        self.updateInterference(arg[1])

    # Override SimPy Resource's "_release" function to update SIR for all incoming messages.
    def _release(self, arg):
        # "arg[1] is a reference to the Packet instance that just completed
        packet = arg[1]
        assert isinstance(packet, PHYPacket), \
            "%s is not a PHY Packet" % str(packet)
        doomed = packet.doomed
        minSIR = packet.getMinSIR()

        payload = packet.decap()

        # Reduce the overall interference by this message's power
        self.interference -= packet.power
        # Prevent rounding errors
        #TODO shouldn't this be to <= ambient?
        self.interference = max(self.interference, self.phy.ambient_noise)

        # Delete this from the transducer queue by calling the Parent form of "_release"
        Sim.Resource._release(self, arg)

        # If it isn't doomed due to transmission & it is not interfered
        if minSIR > 0:
            if not doomed \
                and Linear2DB(minSIR) >= self.phy.threshold['SIR'] \
                and packet.power >= self.phy.threshold['receive']:
                # Properly received: enough power, not enough interference
                self.collision = False
                if debug: self.logger.info("PHY Packet Recieved: %s" % packet.data)
                self.layercake.mac.recv(packet.decap())

            elif packet.power >= self.phy.threshold['receive']:
                # Too much interference but enough power to receive it: it suffered a collision
                if self.host.name == payload.next_hop or self.host.name == payload.destination:
                    self.collision = True
                    self.collisions.append(payload)
                    if debug:
                        self.logger.info("PHY Packet Sensed but not Recieved (%s < %s)" % (
                            packet.power,
                            self.phy.threshold['listen']
                        ))

                else:
                    # Not enough power to be properly received: just heard.
                    self.phy.logger.debug("This packet was not addressed to me.")
            else:
                self.phy.logger.debug("Packet Not Recieved")

        else:
            # This should never appear, and in fact, it doesn't, but just to detect bugs (we cannot have a negative SIR in lineal scale).
            print packet.type, packet.source, packet.dest, packet.next_hop, self.physical_layer.host.name


    def onTX(self):
        self.transmitting = True
        # Doom all currently incoming packets to failure.
        [i.Doom() for i in self.activeQ]


    def postTX(self):
        self.transmitting = False


#####################################################################
# Acoustic Event Listener
#####################################################################

class AcousticEventListener(Sim.Process):
    """No physical analog.
    Waits for another node to send something and then activates
    an Arrival Scheduler instance.
    """

    def __init__(self, transducer):
        Sim.Process.__init__(self)
        self.transducer = transducer

    def listen(self, channel_event, position_query):
        while True:
            #Wait until something happens on the channel
            yield Sim.waitevent, self, channel_event

            params = channel_event.signalparam
            sched = ArrivalScheduler(name="ArrivalScheduler" + self.name[-1])
            Sim.activate(sched, sched.schedule_arrival(self.transducer, params, position_query()))


#####################################################################
# Arrival Scheduler
#####################################################################

class ArrivalScheduler(Sim.Process):
    """simulates the transit time of a message
    """

    def schedule_arrival(self, transducer, params, pos):
        packet = params['packet']
        if debug:    transducer.logger.debug("Scheduling Arrival of packet %s from %s at %s" % \
                                             (packet.data,
                                              packet.source,
                                              pos)
        )
        distance_to = distance(pos, params['pos'])

        if distance_to > 0.01:  # I should not receive my own transmissions
            attenuation_loss = Attenuation(params["frequency"], distance_to)

            receive_power_db = Linear2DB(params["power"]) - attenuation_loss
            travel_time = distance_to / speed_of_sound  # Speed of sound in water = 1482.0 m/s

            receive_power = DB2Linear(receive_power_db)

            transducer.logger.debug("Packet from %s to %s will take %s to cover %s" % (
            packet.source, packet.destination, travel_time, distance_to))

            yield Sim.hold, self, travel_time

            if debug:
                transducer.logger.debug("Scheduled arrival of Packet :%s with power %s will take %s" % (
                    packet.data, receive_power, params["duration"]))
            new_incoming_packet = PHYPacket(packet=packet, power=receive_power)
            Sim.activate(new_incoming_packet,
                         new_incoming_packet.recv(transducer=transducer, duration=params["duration"]))
        else:
            transducer.logger.debug("Transmission too close")


