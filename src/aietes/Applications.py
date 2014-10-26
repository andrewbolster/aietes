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

from numpy import *
from numpy.random import poisson

from collections import Counter
import networkx as nx

from aietes.Tools import Sim, debug, randomstr, broadcast_address


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
                      'packets_received': 0,
                      'packets_time': 0,
                      'packets_hops': 0,
                      'packets_dhops': 0
        }
        self.packet_log = {}
        self.config = config
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

    def tick(self):
        """ Method called at each simulation instance"""
        pass

    def lifecycle(self, destination=None):

        if destination is None:
            destination = self.default_destination

        while True:
            (packet, period) = self.packetGen(period=self.period,
                                              data=randomstr(24),
                                              destination=destination)
            if packet is not None:
                if debug: self.logger.debug("Generated Payload %s: Waiting %s" % (packet.data, period))
                self.layercake.send(packet)
                self.stats['packets_sent'] += 1
            yield Sim.hold, self, period

    def recv(self, FromBelow):
        """
        Called by RoutingTable on packet reception
        """
        self.logPacket(FromBelow)
        self.packetRecv(FromBelow)

    def logPacket(self, packet):
        """
        Grab packet statistics
        """
        source = packet["route"][0][0]
        self.log.append(packet)
        if debug:
            self.logger.info("App Packet received from %s" % source)
        self.stats['packets_recieved'] += 1
        if source in self.packet_log.keys():
            self.packet_log[source].append(packet)
        else:
            self.packet_log[source] = [packet]
        delay = Sim.now() - packet['initial_time']
        # Ignore first hop (source)
        hops = len(packet['route'])

        self.stats['packets_time'] += delay
        self.stats['packets_hops'] += hops
        self.stats['packets_dhops'] += (delay / hops)

        self.logger.debug("Packet received from %s over %d hops with a delay of %s (d/h=%s)" % (
            source, hops, str(delay), str(delay / hops)))

    def dump_stats(self):
        return {
            "rx_counts":self.stats['packets_received'],
            "tx_counts":self.stats['packets_sent'],
            "delays":self.stats['packets_time'],
            "hops":self.stats['packets_hops'],
            "dhops":self.stats['packets_dhops'],
        }

    def packetGen(self, period, destination, *args, **kwargs):
        """
        Packet Generator with periodicity
        Called from the lifecycle with defaults None,None
        """
        raise TypeError("Tried to instantiate the base Application class")


class AccessibilityTest(Application):
    default_destination= broadcast_address

    def packetGen(self, period, destination, data=None, *args, **kwargs):
        """
        Copy of behaviour from AUVNetSim for default class,
        exhibiting poisson departure behaviour
        """
        packet_ID =  self.layercake.hostname+str(self.stats['packets_sent'])
        packet = {"ID": packet_ID, "dest": destination, "source": self.layercake.hostname, "route": [], "type": "DATA", "initial_time": Sim.now(), "length": self.node.config["DataPacketLength"]}
        period = poisson(float(period))
        return packet, period

    def packetRecv(self, packet):
        delay = Sim.now()-packet["initial_time"]
        hops = len(packet["route"])

        self.logger.warn("Packet "+packet["ID"]+" received over "+str(hops)+" hops with a delay of "+str(delay)+"s (delay/hop="+str(delay/hops)+").")




class RoutingTest(Application):

    default_destination= None

    def __init__(self,*args,**kwargs):
        super(RoutingTest,self).__init__(*args,**kwargs)
        self.sent_counter = Counter()
        self.recieved_counter = Counter()
        self.total_counter = Counter()
        self.graph = nx.Graph()

    def packetGen(self, period, destination=None, data=None, *args, **kwargs):
        """
        Lowest-count node gets a message
        """
        if destination is not None:
            raise RuntimeWarning("This isn't the kind of application you use with a destination bub")

        # Update packet_counters with information from the routing layer
        indirect_nodes = filter(lambda n: n not in self.total_counter.keys(), self.layercake.net.keys())
        if len(indirect_nodes):
            self.logger.warning("Inferred new nodes: {}".format(indirect_nodes))
            for node in indirect_nodes:
                self.total_counter[node]=0

        self.mergeCounters()

        if len(self.total_counter):
            most_common = self.total_counter.most_common()
            destination,count = most_common[-1]
            self.sent_counter[destination]+=1
            self.logger.info("Sending to {} with count {}({})".format(destination, count, most_common))
        else:
            self.logger.warn("No Packet Count List set up yet; fudging it with an broadcast first")
            destination=broadcast_address


        packet_ID =  self.layercake.hostname+str(self.stats['packets_sent'])
        packet = {"ID": packet_ID, "dest": destination, "source": self.layercake.hostname, "route": [], "type": "DATA", "initial_time": Sim.now(), "length": self.config["packet_length"]}
        period = poisson(float(period))
        return packet, period

    def packetRecv(self, packet):
        self.mergeCounters()
        self.recieved_counter[packet['source']]+=1
        del packet

    def mergeCounters(self):
        self.total_counter = self.sent_counter + self.recieved_counter
        not_in_rx = filter(lambda n: n not in self.recieved_counter.keys(), self.total_counter.keys())
        not_in_tx = filter(lambda n: n not in self.sent_counter.keys(), self.total_counter.keys())
        if not_in_rx or not_in_tx:
            self.logger.info("Synchronising counters: {} not in rx and {} not in tx".format(not_in_rx, not_in_tx))
        for n in not_in_rx:
            self.recieved_counter[n]=0
        for n in not_in_tx:
            self.sent_counter[n]=0

    def dump_stats(self):
        initial = Application.dump_stats(self)
        initial.update({
            'sent_counts':self.sent_counter,
            'recieved_counts':self.recieved_counter,
            'total_counts':self.total_counter
        })
        return initial

CommsTrust = AccessibilityTest
# class CommsTrust(Application):
#     Trust = collections.namedtuple('Trust', ['plr','rssi','delay','throughput'])
#     my_trust = Trust()
#     network_trust = []
#     def tick(self):
#         pass
#     def packetGen(self, period, destination, *args, **kwargs):
#         data =

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
