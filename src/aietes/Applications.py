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

from collections import Counter, OrderedDict
import networkx as nx

from aietes.Tools import Sim, debug, randomstr, broadcast_address, ConfigError


debug = False


class Application(Sim.Process):
    """
    Generic Class for top level application layers
    """
    HAS_LAYERCAKE = True
    random_delay = False

    def __init__(self, node, config, layercake):
        self._start_log(node)
        Sim.Process.__init__(self)
        self.stats = {'packets_sent': 0,
                      'packets_received': 0,
                      'packets_time': 0,
                      'packets_hops': 0,
                      'packets_dhops': 0
        }
        self.received_log = OrderedDict()
        self.sent_log = []
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
            raise ConfigError("Packet Rate/Count doesn't make sense! {}/{}".format(
                packet_rate,
                packet_count
            ))

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

        if self.random_delay:
            yield Sim.hold, self, poisson(self.random_delay)

        while True:
            (packet, period) = self.packetGen(period=self.period,
                                              data=randomstr(24),
                                              destination=destination)
            if packet is not None:
                if debug: self.logger.debug("Generated Payload %s: Waiting %s" % (packet['data'], period))
                self.layercake.send(packet)
                self.stats['packets_sent'] += 1
                self.sent_log.append(packet)
            yield Sim.hold, self, period

    def recv(self, FromBelow):
        """
        Called by RoutingTable on packet reception
        """
        if debug: self.logger.info("Got Packet {id} from {src}".format(
            id=FromBelow['ID'],
            src=FromBelow['source']
        ))
        self.logPacket(FromBelow)
        self.packetRecv(FromBelow)

    def logPacket(self, packet):
        """
        Grab packet statistics
        """
        source = packet["route"][0][0]
        if hasattr(self.received_log,source):
            self.received_log[source].append(packet)
        else:
            self.received_log[source]=[packet]

        if debug:
            self.logger.info("App Packet received from %s" % source)
        self.stats['packets_received'] += 1
        if source in self.received_log.keys():
            self.received_log[source].append(packet)
        else:
            self.received_log[source] = [packet]
        delay = Sim.now() - packet['initial_time']
        # Ignore first hop (source)
        hops = len(packet['route'])

        self.stats['packets_time'] += delay
        self.stats['packets_hops'] += hops
        self.stats['packets_dhops'] += (delay / hops)

        self.logger.debug("Packet received from %s over %d hops with a delay of %s (d/h=%s)" % (
            source, hops, str(delay), str(delay / hops)))

    def dump_stats(self):


        total_bits_in = total_bits_out = 0
        for txr, pkts in self.received_log.items():
            for pkt in pkts:
                total_bits_in += pkt['length']

        for pkt in self.sent_log:
            total_bits_out += pkt['length']

        throughput = total_bits_in/Sim.now()*self.stats['packets_hops']
        offeredload = total_bits_out/Sim.now()*self.stats['packets_hops']
        try:
            avg_length = total_bits_in / self.stats['packets_received']
        except ZeroDivisionError:
            if self.stats['packets_received'] == 0:
                avg_length = 0
            else:
                raise RuntimeError("Got a zero in a weird place: {}/{}".format(
                    total_bits_in,
                    self.stats['packets_received']
                ))

        left_in_q= len(self.layercake.mac.outgoing_packet_queue)

        app_stats = {
            "rx_counts": self.stats['packets_received'],
            "tx_counts": self.stats['packets_sent'],
            "delays": self.stats['packets_time'],
            "hops": self.stats['packets_hops'],
            "dhops": self.stats['packets_dhops'],
            "average_length": avg_length,
            "throughput": throughput,
            "offeredload": offeredload,
            "enqueued": left_in_q
        }
        app_stats.update(self.layercake.phy.dump_stats())
        return app_stats

    def packetGen(self, period, destination, *args, **kwargs):
        """
        Packet Generator with periodicity
        Called from the lifecycle with defaults None,None
        """
        raise TypeError("Tried to instantiate the base Application class")


class AccessibilityTest(Application):
    default_destination = broadcast_address

    def packetGen(self, period, destination, data=None, *args, **kwargs):
        """
        Copy of behaviour from AUVNetSim for default class,
        exhibiting poisson departure behaviour
        """
        packet_ID = self.layercake.hostname + str(self.stats['packets_sent'])
        packet = {"ID": packet_ID, "dest": destination, "source": self.layercake.hostname, "route": [], "type": "DATA", "initial_time": Sim.now(), "length": self.node.config["DataPacketLength"]}
        period = poisson(float(period))
        return packet, period

    def packetRecv(self, packet):
        delay = Sim.now() - packet["initial_time"]
        hops = len(packet["route"])

        self.logger.warn("Packet " + packet["ID"] + " received over " + str(hops) + " hops with a delay of " + str(delay) + "s (delay/hop=" + str(delay / hops) + ").")


class RoutingTest(Application):
    default_destination = None
    random_delay = 10

    def __init__(self, *args, **kwargs):
        super(RoutingTest, self).__init__(*args, **kwargs)
        self.sent_counter = Counter()
        self.received_counter = Counter()
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
            self.logger.debug("Inferred new nodes: {}".format(indirect_nodes))
            for node in indirect_nodes:
                self.total_counter[node] = 0

        self.mergeCounters()

        if len(self.received_counter)>1:
            most_common = self.received_counter.most_common()
            least_count = most_common[-1][1]
            destination = random.choice([n for n,c in most_common if c == least_count])
            self.sent_counter[destination] += 1
            self.logger.info("Sending to {} with count {}({})".format(destination, least_count, most_common))
        elif len(self.total_counter) == 1:
            destination = self.total_counter.keys()[0]
            self.sent_counter[destination] += 1
            self.logger.info("Sending to {} as it's the only one we know".format(destination))
        else:
            self.logger.warn("No Packet Count List set up yet; fudging it with an broadcast first")
            destination = broadcast_address

        packet_ID = self.layercake.hostname + str(self.stats['packets_sent'])
        packet = {"ID": packet_ID,
                  "dest": destination,
                  "source": self.layercake.hostname,
                  "route": [], "type": "DATA",
                  "initial_time": Sim.now(),
                  "length": self.config["packet_length"],
                  "data": None}
        period = poisson(float(period))
        return packet, period

    def packetRecv(self, packet):
        self.mergeCounters()
        self.received_counter[packet['source']] += 1
        del packet

    def mergeCounters(self):
        learned_nodes = self.sent_counter.keys() + self.received_counter.keys()
        learned_and_implied_nodes = set(self.total_counter.keys()) | set(learned_nodes)
        not_in_rx = filter(lambda n: n not in self.received_counter.keys(), learned_and_implied_nodes)
        not_in_tx = filter(lambda n: n not in self.sent_counter.keys(), learned_and_implied_nodes)
        not_in_tot = filter(lambda n: n not in self.total_counter.keys(), learned_and_implied_nodes)
        for n in not_in_rx:
            self.received_counter[n] = 0
        for n in not_in_tx:
            self.sent_counter[n] = 0
        for n in learned_and_implied_nodes:
            self.total_counter[n] = self.sent_counter[n]+self.received_counter[n]
        if not_in_rx or not_in_tx or not_in_tot:
            self.logger.info("Synchronising counters: {} not in rx and {} not in tx, {} not in total".format(not_in_rx, not_in_tx, not_in_tot))

    def dump_stats(self):
        initial = Application.dump_stats(self)
        initial.update({
            'sent_counts': frozenset(self.sent_counter.items()),
            'received_counts': frozenset(self.received_counter.items()),
            'total_counts': frozenset(self.total_counter.items())
        })
        return initial

class CommsTrust(RoutingTest):
    """
    Vaguely Emulated Bellas Traffic Scenario
    """
    current_target=None
    test_stream_length=10
    stream_period_ratio = 0.1

    def activate(self):
        self.forced_nodes = self.layercake.host.fleet.nodeNames()
        if self.forced_nodes:
            for node in self.forced_nodes:
                if node != self.layercake.hostname:
                    self.total_counter[node]=0
        self.test_packet_counter = Counter(self.total_counter)
        self.result_packet_dl = { name: [] for name in self.total_counter.keys() }
        super(CommsTrust,self).activate()

    def tick(self):
        pass

    def packetGen(self, period, destination=None, data=None, *args, **kwargs):
        """
        Lowest-count node gets a message
        """
        if destination is not None:
            raise RuntimeWarning("This isn't the kind of application you use with a destination bub")

        # Update packet_counters with information from the routing layer
        indirect_nodes = filter(lambda n: n not in self.total_counter.keys(), self.layercake.net.keys())
        if len(indirect_nodes):
            self.logger.debug("Inferred new nodes: {}".format(indirect_nodes))
            for node in indirect_nodes:
                self.total_counter[node] = 0

        # DOES NOT MERGE TARGET COUNTER
        self.mergeCounters()

        most_common = self.total_counter.most_common()
        if not self.current_target or self.test_packet_counter[self.current_target] < self.test_stream_length:
            most_common = self.test_packet_counter.most_common()
            least_count = most_common[-1][1]
            destination = self.current_target = random.choice([n for n,c in most_common if c == least_count])
            self.test_packet_counter[self.current_target] += 1
            self.sent_counter[destination] += 1
            self.logger.info("Sending test packet {} to {} with count ({})".format(self.test_packet_counter[destination],destination, most_common))
        else:
            destination = self.current_target
            self.test_packet_counter+=1
            self.sent_counter[destination] += 1
            self.logger.info("Sending test packet {} to {} with count ({})".format(self.test_packet_counter,destination, most_common))
            if self.test_packet_counter[self.current_target] >= self.test_stream_length:
                self.current_target=None
                period = poisson(float(period))
                self.logger.warn("LAST PACKET FOR {}".destination)
            else:
                period = poisson(float(period*self.stream_period_ratio))

        packet_ID = self.layercake.hostname + str(self.stats['packets_sent'])
        packet = {"ID": packet_ID,
                  "dest": destination,
                  "source": self.layercake.hostname,
                  "route": [], "type": "DATA",
                  "initial_time": Sim.now(),
                  "length": self.config["packet_length"],
                  "data": self.test_packet_counter[destination]}
        period = poisson(float(period))
        return packet, period

    def packetRecv(self, packet):
        self.mergeCounters()
        self.received_counter[packet['source']] += 1
        self.result_packet_dl[packet['source']].append(packet['data'])
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
