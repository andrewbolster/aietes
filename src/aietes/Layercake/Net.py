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

import collections
import operator

from Packet import RoutePacket, AppPacket
from aietes.Tools import debug, Sim, broadcast_address

debug = False


class NetLayer(object):
    """Routing table generic class
    """
    has_routing_table = False

    def __init__(self, layercake, config=None):
        # Generic Spec
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)
        if debug: self.logger.debug('creating instance:%s' % config)

        self.layercake = layercake
        self.host = layercake.host
        self.config = config
        self.table = {}
        self.packets = dict()

    def send(self, FromAbove):
        # Take Application Packet
        packet = RoutePacket(self, FromAbove)

        if not hasattr(self.table, packet.destination):
            packet.set_next_hop(packet.destination)
        else:
            packet.set_next_hop(self.table[packet.destination])
        if debug: self.logger.debug("Net Packet, %s, sent to %s" % (packet.data, packet.next_hop))
        self.layercake.mac.send(packet)


    def recv(self, FromBelow):
        packet = FromBelow.decap()

        # IF it's for us, send it up to the app layer, if not, send if back
        if not self.hasDuplicate(packet):
            self.packets[packet.id] = packet
            if packet.next_hop == packet.destination:
                if packet.isFor(self.host):
                    self.push_up(packet)
                else:
                    raise RuntimeError("WTFMATE?")
            else:
                self.logger.error("Don't know what to do with packet " + packet.data + " from " + \
                                  packet.source + " going to " + packet.destination + " with hop " + packet.next_hop)
                raise NotImplemented
        else:
            self.logger.debug("Already Have Pkt {}:{}".format(packet.data, packet.id))

    def push_up(self, packet):
        """
        Commonly overriden 'final step' for all packets that are FOR this node (i.e me or broadcast)
        :param packet:
        :return:
        """
        self.layercake.recv(packet)

    def explicitACK(self, FromBelow):
        """Assume we always want to call for ACK
        i.e. no implicit ACK
        """
        if self.hasDuplicate(FromBelow):
            return False
        if FromBelow.type is "ACK":
            return False
        elif FromBelow.requests_ack:
            return False
        else:
            return True

    def hasDuplicate(self, packet):
        """ Checks if the packet has already been dealt with"""
        if packet.id in self.packets.keys():
            return True
        else:
            return False


class DSDV(NetLayer, Sim.Process):
    """
    Destination Sequenced Discance Vector protocol uses the Bellnman Ford Algo. to calculate paths based on hop-lengths

    Maintains two routing tables, one permemant and one advertised (i.e. stable and unstable)

    """
    has_routing_table = True  # NA
    periodic_update_interval = 15  # Time between full-table exchanges among nodes
    wst_enabled = True  # Enables Weighted Settling Time for updates before Advertisment
    settling_time = 6  # Minimum storage time before tx
    wst_factor = 0.875  # Fairly Obvious....
    buffer_enabled = True  # Buffer if no rout available
    max_queue_len = 100  # Obvious
    max_queue_time = 30  # Obvious
    max_queued_per_dest = 5  # Q / Destination
    hold_times = 3  # how many periodic updates before purging a route
    route_agg_enabled = False  # Aggregated updates
    route_agg_time = 1  # seconds over which DSDV updates are aggregated

    RoutingEntry = collections.namedtuple("RoutingEntry",
                                          ["next", "hops", "seq"])

    class DSDV_Packet(AppPacket):
        """
        Packet Mask to simulate header encapsulation
        """

        def __init__(self, source, table):
            AppPacket.__init__(self, source, dest=broadcast_address, pkt_type="DATA", data=table, route=None, logger=None)
            if table is None:
                self.logger.error("Shouldn't you be sending a table?")

            self.requests_ack = False

    def __init__(self, layercake, config=None):
        NetLayer.__init__(self, layercake, config)
        Sim.Process.__init__(self, name=self.__class__.__name__)
        self.seq_no = reduce(operator.xor, map(ord, self.host.name)) * 100  # Sneaky trick to make 'recognisable' sequence numbers'
        self.table = {}

    def push_up(self, packet):
        if not self.active():  # inherited from Process
            Sim.activate(self, self.lifecycle())
            self.logger.info("Activated Lifecycle on first packet reception")

        if isinstance(packet.payload, DSDV.DSDV_Packet):
            self.process_update_from_packet(packet)
        else:
            self.layercake.recv(packet)

    def process_update_from_packet(self, packet):
        """
        Update the Routing Table based on incoming info from the packet
        :param packet:
        :return:
        """
        for dest, hop, seq_no in packet.payload.data:
            if not hasattr(self.table, dest):
                new_entry = DSDV.RoutingEntry(packet.source, hop, seq_no)
                self.logger.info("Creating entry for {}:{}".format(dest, new_entry))
                self.table[dest] = new_entry
            elif seq_no > self.table[dest].seq:
                new_entry = DSDV.RoutingEntry(packet.source, hop, seq_no)
                self.logger.info("Updating entry for {}:{}".format(dest, new_entry))
                self.table[dest] = new_entry

    def broadcast_table(self):

        self.seq_no += 2
        tx_table = [(self.host.name, 0, self.seq_no)]
        for dest, entry in self.table.items():
            tx_table.append((dest, entry.hops, entry.seq))

        self.logger.info("Broadcasting Table {}".format(tx_table))

        self.send(
            DSDV.DSDV_Packet(
                source=self.host.name, table=tx_table
            )
        )


    def lifecycle(self):
        while True:
            self.broadcast_table()
            yield Sim.hold, self, self.periodic_update_interval




