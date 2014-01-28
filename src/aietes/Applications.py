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

from numpy import *
from numpy.random import poisson
from aietes.Tools import Sim, debug, randomstr, broadcast_address

from Layercake.Packet import AppPacket

#debug=True
class Application(Sim.Process):
    """
    Generic Class for top level application layers
    """
    HAS_LAYERCAKE = True

    def __init__(self, node, config, layercake):
        self._start_log(node)
        Sim.Process.__init__(self)
        self.stats = {'packets_sent': 0,
                      'packets_recieved': 0,
                      'packets_time': 0,
                      'packets_hops': 0,
                      'packets_dhops': 0
        }
        self.packet_log = {}
        self.config = config
        self.logger.info(layercake)
        self.layercake = layercake
        self.sim_duration = node.simulation.config.Simulation.sim_duration

        packet_rate = getattr(config, 'packet_rate')
        packet_count = getattr(config, 'packet_count')
        if packet_rate > 0 and packet_count == 0:
            self.packet_rate = getattr(config, 'packet_rate')
            if debug: self.logger.debug("Taking Packet_Rate from config: %s" % self.packet_rate)
        elif packet_count > 0 and packet_rate == 0:
            # If packet count defined, only send our N packets
            self.packet_rate = packet_count / float(self.sim_duration)
            if debug: self.logger.debug("Taking Packet_Count from config: %s" % self.packet_rate)
        else:
            self.packet_rate = 1
            self.logger.warn("This sure is a weird configuration of Packets! Sending at 1pps anyway")
            # raise Exception("Packet Rate/Count doesn't make sense!")

        self.period = 1 / float(self.packet_rate)

    def _start_log(self, parent):
        self.logger = parent.logger.getChild("App:%s" % self.__class__.__name__)

    def activate(self):
        Sim.activate(self, self.lifecycle())

    def lifecycle(self, destination=None):

        if destination is None:
            if debug: self.logger.debug("No Destination defined, defaulting to \"%s\"" % broadcast_address)
            destination = broadcast_address

        while True:
            (packet, period) = self.packetGen(period=self.period,
                                              data=randomstr(24),
                                              destination=destination)
            if packet is not None:
                if debug:
                    self.logger.debug("Generated Payload %s: Waiting %s" % (packet.data, period))
                self.layercake.send(packet)
                self.stats['packets_sent'] += 1
            yield Sim.hold, self, period

    def recv(self, FromBelow):
        """
        Called by RoutingTable on packet reception
        """
        packet = FromBelow.decap()
        self.logPacket(packet)
        self.packetRecv(packet)

    def logPacket(self, packet):
        """
        Grab packet statistics
        """
        assert isinstance(packet, AppPacket)
        source = packet.source
        if debug:
            self.logger.info("App Packet Recieved from %s" % source)
        self.stats['packets_recieved'] += 1
        if source in self.packet_log.keys():
            self.packet_log[source].append(packet)
        else:
            self.packet_log[source] = [packet]
        delay = Sim.now() - packet.launch_time
        # Ignore first hop (source)
        hops = len(packet.route) - 1

        self.stats['packets_time'] += delay
        self.stats['packets_hops'] += hops
        self.stats['packets_dhops'] += (delay / hops)

        self.logger.info("Packet recieved from %s over %d hops with a delay of %s (d/h=%s)" % (
            source, hops, str(delay), str(delay / hops)))

    def packetGen(self, period, destination, *args, **kwargs):
        """
        Packet Generator with periodicity
        Called from the lifecycle with defaults None,None
        """
        raise TypeError("Tried to instantiate the base Application class")


class AccessibilityTest(Application):
    def packetGen(self, period, destination, data=None, *args, **kwargs):
        """
        Copy of behaviour from AUVNetSim for default class,
        exhibiting poisson departure behaviour
        """
        packet = AppPacket(
            source=self.layercake.host.name,
            dest=destination,
            pkt_type='DATA',
            data=data,
            logger=self.logger
        )
        period = poisson(float(period))
        return packet, period

    def packetRecv(self, packet):
        assert isinstance(packet, AppPacket)
        del packet


class Null(Application):
    HAS_LAYERCAKE = False

    def packetGen(self, period, destination, data=None, *args, **kwargs):
        """
        Does Nothing, says nothing
        """
        return None, 1

    def packetRecv(self, packet):
        assert isinstance(packet, AppPacket)
        del packet
