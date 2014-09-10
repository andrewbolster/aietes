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

from Packet import RoutePacket
from aietes.Tools import debug

debug = False


class RoutingTable():
    """Routing table generic class
    """

    def __init__(self, layercake, config=None):
        #Generic Spec
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)
        if debug: self.logger.debug('creating instance:%s' % config)

        self.layercake = layercake
        self.host = layercake.host
        self.config = config
        self.has_routing_table = False
        self.table = {}
        self.packets = set([])

    def send(self, FromAbove):
        #Take Application Packet
        packet = RoutePacket(self, FromAbove)

        if not hasattr(self.table, packet.destination):
            packet.set_next_hop(packet.destination)
        else:
            packet.set_next_hop(self.table[packet.destination])
        if debug: self.logger.debug("Net Packet, %s, sent to %s" % (packet.data, packet.next_hop))
        self.layercake.mac.send(packet)


    def recv(self, FromBelow):
        packet = FromBelow.decap()
        if debug: self.logger.info("Net Packet Recieved:%s" % packet.data)

        #IF it's for us, send it up to the app layer, if not, send if back
        if not self.hasDuplicate(packet):
            self.packets.add(packet.id)
            if packet.next_hop == packet.destination:
                if packet.isFor(self.host):
                    self.layercake.recv(packet)
            else:
                self.logger.error("Don't know what to do with packet " + packet.data + " from " + \
                                  packet.source + " going to " + packet.destination + " with hop " + packet.next_hop)
                raise NotImplemented

    def explicitACK(self, FromBelow):
        """Assume we always want to call for ACK
        i.e. no implicit ACK
        """
        return True

    def hasDuplicate(self, packet):
        """ Checks if the packet has already been dealt with"""
        return packet.id in self.packets

