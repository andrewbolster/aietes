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

"""
Packets have to cope with bi-directional adaptation:
    Downward (eg app->routing) is simple encapsulation, ie the lower level
    packet inherits the dict of the encapsulated packet (which is also stored
    as a direct attribute, i.e rt_pkt.payload

    Upward is a simple case of pulling out the encapsulated value from the
    lower level packet

Encapsupation runs to:
    PHY(MAC(RT(APP)))
"""
import uuid

from aietes.Tools import *


class Packet(object):
    """Base Packet Class
    """

    def __init__(self, packet, ):
        """ Take the upper level packet and encapsulate it as the payload within the new packet
        while maintaining access to all members.

                    | X | payload|
                    |--->---\/---|
                | Y | X | payload|
                | ------>---\/---|
            | Z | Y | X | payload|

        """
        if hasattr(packet, 'logger'):
            self._start_log(logger=packet.logger)
        else:
            self._start_log()

        #Import everything from the upper packet
        for k, v in packet.__dict__.items():
            if debug: self.logger.debug("Got KV: %s %s" % (k, v))
            setattr(self, k, v)
        self.payload = packet         #Overwrite the 'upper' packet payload with the packet

    def _start_log(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger("%s" % self.__class__.__name__)
        else:
            self.logger = logger.getChild("%s" % self.__class__.__name__)

    def decap(self):
        return self.payload #Should return the upper level packet class

    def isFor(self, node):
        """ Boolean if packet directed at node by either
        name or broadcast
        """
        if self.payload.destination == broadcast_address:
            return True
        elif hasattr(node,'name'):
            # Non-invasive way of assessing if it's a node object or not
            return self.payload.destination == node.name
        else:
            return self.payload.destination == node

    def __str__(self):
        return str("To: %s, From: %s, at %d, %s" % (self.destination, self.source, self.launch_time, self.data))

#####################################################################
# Application Packet
#####################################################################
class AppPacket(Packet):
    """Base Packet Class (AKA AppPacket)
    """
    source = None
    destination = None
    type = None
    data = None
    length = 24 # Default Packet Length

    def __init__(self, source, dest, pkt_type=None, data=None, route=None, logger=None):
        self._start_log(logger=logger)
        self.source = source
        self.destination = dest
        self.type = pkt_type
        self.launch_time = Sim.now()
        self.route = route if route is not None else []
        if data is not None:
            self.data = "{}:{}".format(data,source)
            self.length = len(data)
        else:
            self.data = "{}:{}".format(AppPacket.data,source)
            self.length = AppPacket.length

        self.id = uuid.uuid4() #Hopefully unique id

#####################################################################
# Network Packet
#####################################################################
class RoutePacket(Packet):
    """Routing Level Packet
    """

    def __init__(self, route_layer, packet, tx_level=0.0):
        Packet.__init__(self, packet)
        self.tx_level = tx_level
        self.next_hop = None #Analogue to packet['through'] in AUVNetSim
        self.source_position = None
        self.destination_position = None

        self.route.append({'name': route_layer.host.name,
                           'position': route_layer.host.getPos()
        })
        if debug: self.logger.info("Net Packet Route:%s" % self.route)
        self.source_position = route_layer.host.getPos()

    def set_level(self, tx_level):
        self.tx_level = tx_level

    def set_next_hop(self, hop):
        self.next_hop = hop

#####################################################################
# Acknowledgement Packet
#####################################################################
class ACK(Packet):
    """
    Simple Acknowledgement packet + generation in response to a recieved packet

    Overrides init and payload relevant
    """
    length=24
    def __init__(self, node, packet):
        if hasattr(packet, 'logger'):
            self._start_log(logger=packet.logger)
        else:
            self._start_log()

        if not isinstance(packet, MACPacket):
            raise ValueError("Expecting MACPacket, got {}".format(packet.__class__.__name__))

        self.type = "ACK"
        self.source = node.name
        self.source_position = node.getPos()
        self.destination = packet.source
        self.destination_position = packet.source_position
        self.launch_time = Sim.now()
        self.route = packet.route[::-1]
        self.tx_level = packet.tx_level
        self.data = "ACK:{}".format(packet.data)
        self.next_hop = self.route[0]
        self.length = len(self.data)


####################################################################
# Media Access Control Packet
#####################################################################
class MACPacket(Packet):
    """MAC level packet
    """

    def last_sender(self):
        return self.route[-1]

    def customBehaviour(self):
        #TODO What does a MAC packet do differently?
        pass

#####################################################################
# Physical Packet
#####################################################################
class PHYPacket(Sim.Process, Packet):
    """Generic Physical Layer Packet Class
    Executed as a process and actually performs the send and recieve functions
    for the physical layer.
    This is because the transducer and modem are RESOURCES that are used, not
    processing units.
    """

    def __init__(self, packet, power):
        """
        :param packet: Upper level packet being encapsulated
        :param power: linear power (mW)
        :return:
        """
        Packet.__init__(self, packet)
        Sim.Process.__init__(self, name=self.__class__.__name__)
        self.doomed = False
        self.max_interference = None

        # POWER is TX power initially on encapsulation of the packet, but is the attenuated received power on decapsulation
        #For Incoming Packet (RX), is Linear
        self.power = power


    def send(self, phy):
        assert phy is not None, "Really Buggered your testing, haven't you!"

        self.logger.debug("Sending Packet :%s with power %s" % (self.data, self.power))
        #Lock Modem
        yield Sim.request, self, phy.modem

        #Generate Bandwidth info
        if phy.variable_bandwidth:
            distance = phy.var_power['levelToDistance'][self.packet.tx_level] #TODO Packet description
            bandwidth = distance2Bandwidth(self.power,
                                           phy.frequency,
                                           distance,
                                           phy.threshold['SNR']
            )
        else:
            bandwidth = phy.bandwidth

        bitrate = phy.bandwidth_to_bit(bandwidth)
        duration = self.length / bitrate

        phy.transducer.channel_event.signal({"pos": self.source_position,
                                             "power": self.power,
                                             "duration": duration,
                                             "frequency": phy.frequency,
                                             "speed": phy.medium_speed,
                                             "packet": self
        })

        #Lock the transducer for duration of TX
        phy.transducer.onTX()
        yield Sim.hold, self, duration
        phy.transducer.postTX()

        #Release modem
        yield Sim.release, self, phy.modem

        #Update power stats
        power_w = DB2Linear(AcousticPower(self.power))
        phy.tx_energy += (power_w * duration)


    def recv(self, transducer, duration):
        if self.power >= transducer.threshold['listen']:
            # Otherwise I will not even notice that there are packets in the network
            yield Sim.request, self, transducer
            yield Sim.hold, self, duration
            yield Sim.release, self, transducer

            self.logger.debug("RX from %s with power %.2fdB: %s" % (
                self.source, Linear2DB(self.power), self.data))
            # Even if a packet is not received properly, we have consumed power
            transducer.updatePowerOnRecieve(duration)
        else:
            self.logger.debug("RX from %s with power %.2fdB below lis" % (
            self.source, self.power))

    def updateInterference(self, interference):
        """A simple ratchet of interference based on the transducer _request func
        """
        self.max_interference = max(self.max_interference, interference)

    def getMinSIR(self):
        return self.power / (self.max_interference - self.power + 1) # TODO BODMAS?

    def Doom(self):
        self.doomed = True









