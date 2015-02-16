#!/usr/bin/env python
# coding=utf-8
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

from collections import Counter,namedtuple

import numpy as np
from numpy.random import poisson

import pandas as pd

from aietes.Tools import Sim, DEBUG, randomstr, broadcast_address, ConfigError


DEBUG = False


class Application(Sim.Process):
    """
    Generic Class for top level application layers
    """
    HAS_LAYERCAKE = True
    random_delay = False

    def __init__(self, node, config, layercake):
        self._start_log(node)
        Sim.Process.__init__(self, name= "{}({})".format(
            self.__class__.__name__,
            node.name)
        )
        self.stats = {'packets_sent': 0,
                      'packets_received': 0,
                      'packets_time': 0,
                      'packets_hops': 0,
                      'packets_dhops': 0,
        }
        self.received_log = []
        self.last_accessed_rx_packet = None
        self.sent_log = []
        self.config = config
        self.layercake = layercake
        self.sim_duration = node.simulation.config.Simulation.sim_duration
        if self.HAS_LAYERCAKE:
            self.packet_length = layercake.packet_length
            try:
                packet_rate = self.config['packet_rate']
                packet_count = self.config['packet_count']
            except:
                self.logger.error(config)
                raise

            if packet_rate > 0 and packet_count == 0:
                self.packet_rate = packet_rate
                if DEBUG:
                    self.logger.debug(
                        "Taking Packet_Rate from config: %s" % self.packet_rate)
            elif packet_count > 0 and packet_rate == 0:
                # If packet count defined, only send our N packets
                self.packet_rate = packet_count / float(self.sim_duration)
                if DEBUG:
                    self.logger.debug(
                        "Taking Packet_Count from config: %s" % self.packet_rate)
            else:
                raise ConfigError("Packet Rate/Count doesn't make sense! {}/{}".format(
                    packet_rate,
                    packet_count
                ))

            self.period = 1 / float(self.packet_rate)
        else:
            self.period = 0

    def _start_log(self, parent):
        self.logger = parent.logger.getChild(
            "App:%s" % self.__class__.__name__)

    def activate(self):

        if self.HAS_LAYERCAKE:
            self.layercake.tx_good_signal_hdlrs.append(self.signal_good_tx)
            self.layercake.tx_lost_signal_hdlrs.append(self.signal_lost_tx)

        Sim.activate(self, self.lifecycle())


    def signal_good_tx(self, packetid):
        for sent in self.sent_log:
            if sent['ID'] == packetid:
                sent['delivered'] = True
                sent['acknowledged'] = Sim.now()

    def signal_lost_tx(self, packetid):
        for sent in self.sent_log:
            if sent['ID'] == packetid:
                sent['delivered'] = False
                sent['acknowledged'] = Sim.now()

    def tick(self):
        """ Method called at each simulation instance"""
        pass

    def lifecycle(self, destination=None):

        if destination is None:
            destination = self.default_destination

        if self.random_delay:
            yield Sim.hold, self, poisson(self.random_delay)

        while True:
            (packet, period) = self.generate_next_packet(period=self.period,
                                              data=randomstr(24),
                                              destination=destination)
            if packet is not None:
                if DEBUG:
                    self.logger.debug(
                        "Generated Payload %s: Waiting %s" % (packet['data'], period))
                self.layercake.send(packet)
                self.stats['packets_sent'] += 1
                packet['delivered'] = None
                self.sent_log.append(packet)
            yield Sim.hold, self, period

    def recv(self, from_below):
        """
        Called by RoutingTable on packet reception
        """
        if DEBUG:
            self.logger.info("Got Packet {id} from {src}".format(
                id=from_below['ID'],
                src=from_below['source']
            ))
        from_below['received'] = Sim.now()
        self.log_received_packet(from_below)
        self.packet_recv(from_below)

    def packet_recv(self, packet):
        pass

    def log_received_packet(self, packet):
        """
        Grab packet statistics
        """
        source = packet["route"][0][0]

        if DEBUG:
            self.logger.info("App Packet received from %s" % source)
        self.stats['packets_received'] += 1

        delay = packet['delay'] = packet['received'] - packet['time_stamp']

        self.received_log.append(packet)

        # Ignore first hop (source)
        hops = len(packet['route'])

        self.stats['packets_time'] += delay
        self.stats['packets_hops'] += hops
        self.stats['packets_dhops'] += (delay / hops)

        self.logger.debug("Packet received from %s over %d hops with a delay of %s (d/h=%s)" % (
            source, hops, str(delay), str(delay / hops)))

    def dump_stats(self):

        """


        :return: :raise RuntimeError:
        """
        if self.HAS_LAYERCAKE:
            total_bits_in = total_bits_out = 0
            for pkt in self.received_log:
                total_bits_in += pkt['length']

            for pkt in self.sent_log:
                total_bits_out += pkt['length']

            throughput = total_bits_in / Sim.now() * self.stats['packets_hops']
            offeredload = total_bits_out / Sim.now() * self.stats['packets_hops']
            try:
                avg_length = total_bits_in / self.stats['packets_received']
                average_rx_delay = self.stats['packets_time'] / self.stats['packets_received']
            except ZeroDivisionError:
                if self.stats['packets_received'] == 0:
                    avg_length = 0
                    average_rx_delay = np.inf
                else:
                    raise RuntimeError("Got a zero in a weird place: {}/{}".format(
                        total_bits_in,
                        self.stats['packets_received']
                    ))

            left_in_q = len(self.layercake.mac.outgoing_packet_queue)
            try:
                rts_count = self.layercake.mac.total_channel_access_attempts
            except AttributeError:
                rts_count = np.nan

            app_stats = {
                "rx_counts": self.stats['packets_received'],
                "tx_counts": self.stats['packets_sent'],
                "rts_counts": rts_count,
                "average_rx_delay": average_rx_delay,
                "delay": self.stats['packets_time'],
                "hops": self.stats['packets_hops'],
                "dhops": self.stats['packets_dhops'],
                "average_length": avg_length,
                "throughput": throughput,
                "offeredload": offeredload,
                "enqueued": left_in_q
            }
            app_stats.update(self.layercake.phy.dump_stats())
        else:
            app_stats = {}
        return app_stats


    def dump_logs(self):
        """
        Return the packet tx/rx logs
        """
        if self.HAS_LAYERCAKE:
            return {
                'tx': self.sent_log,
                'rx': self.received_log,
                'tx_queue': self.layercake.mac.outgoing_packet_queue
            }
        else:
            return {}

    def generate_next_packet(self, period, destination, *args, **kwargs):
        """
        Packet Generator with periodicity
        Called from the lifecycle with defaults None,None
        :param period:
        :param destination:
        :param args:
        :param kwargs:
        """
        raise TypeError("Tried to instantiate the base Application class")


class AccessibilityTest(Application):
    default_destination = broadcast_address

    def generate_next_packet(self, period, destination, data=None, *args, **kwargs):
        """
        Copy of behaviour from AUVNetSim for default class,
        exhibiting poisson departure behaviour
        :param period:
        :param destination:
        :param data:
        :param args:
        :param kwargs:
        """
        packet_id = self.layercake.hostname + str(self.stats['packets_sent'])
        packet = {"ID": packet_id, "dest": destination, "source": self.layercake.hostname, "route": [
        ], "type": "DATA", "time_stamp": Sim.now(), "length": self.packet_length}
        period = poisson(float(period))
        return packet, period

    def packet_recv(self, packet):
        """

        :param packet:
        """
        delay = Sim.now() - packet["time_stamp"]
        hops = len(packet["route"])

        self.logger.warn("Packet " + packet["ID"] + " received over " + str(
            hops) + " hops with a delay of " + str(delay) + "s (delay/hop=" + str(delay / hops) + ").")


class RoutingTest(Application):
    """

    :param args:
    :param kwargs:
    """
    default_destination = None
    random_delay = 10

    def __init__(self, *args, **kwargs):
        super(RoutingTest, self).__init__(*args, **kwargs)
        self.sent_counter = Counter()
        self.received_counter = Counter()
        self.total_counter = Counter()

    def generate_next_packet(self, period, destination=None, data=None, *args, **kwargs):
        """
        Lowest-count node gets a message
        :param period:
        :param destination:
        :param data:
        :param args:
        :param kwargs:
        """
        if destination is not None:
            raise RuntimeWarning(
                "This isn't the kind of application you use with a destination bub")

        # Update packet_counters with information from the routing layer
        indirect_nodes = filter(
            lambda n: n not in self.total_counter.keys(), self.layercake.net.keys())
        if len(indirect_nodes):
            self.logger.debug("Inferred new nodes: {}".format(indirect_nodes))
            for node in indirect_nodes:
                self.total_counter[node] = 0

        self.merge_counters()

        if len(self.received_counter) > 1:
            most_common = self.received_counter.most_common()
            least_count = most_common[-1][1]
            destination = np.random.choice(
                [n for n, c in most_common if c == least_count])
            self.sent_counter[destination] += 1
            self.logger.info("Sending to {} with count {}({})".format(
                destination, least_count, most_common))
        elif len(self.total_counter) == 1:
            destination = self.total_counter.keys()[0]
            self.sent_counter[destination] += 1
            self.logger.info(
                "Sending to {} as it's the only one we know".format(destination))
        else:
            self.logger.warn(
                "No Packet Count List set up yet; fudging it with an broadcast first")
            destination = broadcast_address

        packet_id = self.layercake.hostname + str(self.stats['packets_sent'])
        packet = {"ID": packet_id,
                  "dest": destination,
                  "source": self.layercake.hostname,
                  "route": [], "type": "DATA",
                  "time_stamp": Sim.now(),
                  "length": self.packet_length,
                  "data": None}
        period = poisson(float(period))
        return packet, period

    def packet_recv(self, packet):
        """

        :param packet:
        """
        self.merge_counters()
        self.received_counter[packet['source']] += 1
        del packet

    def merge_counters(self):
        """


        """
        learned_nodes = self.sent_counter.keys() + self.received_counter.keys()
        learned_and_implied_nodes = set(
            self.total_counter.keys()) | set(learned_nodes)
        not_in_rx = filter(
            lambda n: n not in self.received_counter.keys(), learned_and_implied_nodes)
        not_in_tx = filter(
            lambda n: n not in self.sent_counter.keys(), learned_and_implied_nodes)
        not_in_tot = filter(
            lambda n: n not in self.total_counter.keys(), learned_and_implied_nodes)
        for n in not_in_rx:
            self.received_counter[n] = 0
        for n in not_in_tx:
            self.sent_counter[n] = 0
        for n in learned_and_implied_nodes:
            self.total_counter[n] = self.sent_counter[
                                        n] + self.received_counter[n]
        if not_in_rx or not_in_tx or not_in_tot:
            self.logger.info("Synchronising counters: {} not in rx and {} not in tx, {} not in total".format(
                not_in_rx, not_in_tx, not_in_tot))

    def dump_stats(self):
        """


        :return:
        """
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
    Trust Values retained per interval:
        Packet Loss Rate

    Default Trust Assessment Period of 10 minutes
    Default to Random Delay < 60 s before first cycle - reduces contension.

    """
    test_stream_length = 6
    stream_period_ratio = 0.1
    trust_assessment_period = 600
    random_delay = 60
    metrics_string = "ATXP,ARXP,ADelay,ALength,Throughput,PLR"

    def activate(self):

        """


        """
        self.current_target = None
        self.last_trust_assessment = None
        self.last_accessed_rx_packet = None

        self.trust_assessments = {}  # My generated trust metrics, [node][t][n_observation_arrays]
        # Extra information that might be interesting in the longer term.
        self.trust_accessories = {'queue_length': [],
                                  'collisions': []
        }

        self.forced_nodes = self.layercake.host.fleet.node_names()
        if self.forced_nodes:
            for node in self.forced_nodes:
                if node != self.layercake.hostname:
                    self.total_counter[node] = 0
        self.test_packet_counter = Counter(self.total_counter)
        self.result_packet_dl = {name: []
                                 for name in self.total_counter.keys()}

        super(CommsTrust, self).activate()

    @classmethod
    def get_metrics_from_packet(cls, packet):
        """
        Extracts the following measurements from a packet
            - tx power
            - rx power
            - delay
            - data length
        :param packet:
        :return:
        """
        pkt_indexes_map = {
            'tx_pwr_db':"TXP",
            'rx_pwr_db':"RXP",
            'delay':"Delay",
            'length':"Length"
        }
        return pd.Series({v:packet[k] for k,v in pkt_indexes_map.items()})

    def get_metrics_from_batch(self, batch):
        """
        Extracts per-packet metrics, averages them, and tags on the following cross packet metrics
            - rx-throughput
        :param batch:
        :return:
        """

        nodepktstats = []
        throughput = 0  # SUM Length

        for j_pkt, pkt in enumerate(batch):
            nodepktstats.append(self.get_metrics_from_packet(pkt))
            throughput += pkt['length']

        if throughput:
            try:
                # Bring individual observations into a frame
                df = pd.concat(nodepktstats,axis=1)
                s = df.mean(axis=1)
                # Prepend the keys with "A" to denote average values
                s.index = ["A"+k for k in s.keys()]
                # Append the Throughput to the series and return
                s['RXThroughput']=throughput
                return s
            except:
                self.logger.exception("PKTS:{},TP:{},NANMEAN:{}".format(
                    nodepktstats, throughput, np.nanmean(nodepktstats, axis=0)
                ))
                raise
        else:
            return pd.Series([])

    def tick(self):
        """
        Processing performed periodically depending on the

        """
        if not Sim.now() % self.trust_assessment_period:
            if self.layercake.hostname=='n0' and DEBUG:
                self.layercake.host.fleet.plot_axes_views().savefig('/dev/shm/test.png')

            # Set up the data structures
            for node in self.total_counter.keys():
                # Relevant Packets RX since last interval

                if not self.trust_assessments.has_key(node):
                    self.trust_assessments[node] = []

            relevant_rx_packets = self.received_log[self.last_accessed_rx_packet:]
            self.last_accessed_rx_packet = len(self.received_log)

            rx_stats = {}
            for node in self.trust_assessments.keys():
                pkt_batch = [packet for packet in relevant_rx_packets if packet['source'] == node]
                rx_stats[node] = self.get_metrics_from_batch(pkt_batch)
                # avg(tx pwr, rx pwr, delay, length), rx throughput

            Pkt=namedtuple('Pkt',['n','dest','length','delivered'])
            last_relevant_time = Sim.now() - self.trust_assessment_period


            # Only concern yourself with packets actually sent out (i.e. made it beyond the queue)
            acked_packets = filter(lambda d: d.has_key("acknowledged"), self.sent_log)
            relevant_acked_packets = []
            for i, p in enumerate(acked_packets):
                if p['acknowledged'] > last_relevant_time:
                    relevant_acked_packets.append(Pkt(i, p['dest'], p['length'], p['delivered']))

            tx_stats = {}
            # Estimate the Packet Error Rate based on the percentage of lost packets sent in the last assessment period
            for node in self.trust_assessments.keys():
                # Throughput is the total length of data transmitted this frame
                tx_throughput = map(
                    lambda p: float(p.length),
                    filter(
                        lambda p: p.dest == node,
                        relevant_acked_packets)
                )
                if tx_throughput:
                    tx_throughput = np.sum(tx_throughput)
                else:
                    tx_throughput = 0.0

                # PLR is the average amount of packets we are sure have been lost in the last time frame
                # Including pkts that have been acked since last time.
                plr = map(
                    lambda p: float(not p.delivered),
                    filter(
                        # We know the fate of all acked packets
                        lambda p: p.dest == node,
                        relevant_acked_packets)
                )
                if plr and tx_throughput>0.0:
                    plr = np.nanmean(plr)
                else:
                    plr = 0.0

                tx_stats[node] = pd.Series({
                    'PLR':plr if not np.isnan(plr) else 0.0,
                    'TXThroughput':tx_throughput
                })

            for node in self.trust_assessments.keys():
                nodestat = pd.concat([rx_stats[node], tx_stats[node]])
                # avg(tx pwr, rx pwr, delay, length), rx throughput, PER
                self.trust_assessments[node].append(nodestat)

            self.trust_accessories['queue_length'].append(len(self.layercake.mac.outgoing_packet_queue))
            self.trust_accessories['collisions'].append(len(self.layercake.phy.transducer.collisions))


    def select_target(self):
        """
        Selects a new target based on the least communicated with

        :return: str
        """
        most_common = self.total_counter.most_common()
        new_target = np.random.choice(
            [n
             for n, c in most_common
             if c == most_common[-1][1]
            ]
        )
        return new_target

    def generate_next_packet(self, period, destination=None, data=None, *args, **kwargs):
        """
        Lowest-count node gets a message indicating what number packet it is that
        is addressed to it with a particular stream length.
        The counter counts the 'actual' number of packets while the packet.data
        carries the zero-indexed 'packet id'
        :param period:
        :param destination:
        :param data:
        :param args:
        :param kwargs:
        """
        if destination is not None:
            raise RuntimeWarning(
                "This isn't the kind of application you use with a destination bub")

        # Update packet_counters with information from the routing layer
        indirect_nodes = filter(
            lambda n: n not in self.total_counter.keys(), self.layercake.net.keys())
        if len(indirect_nodes):
            self.logger.debug("Inferred new nodes: {}".format(indirect_nodes))
            for node in indirect_nodes:
                self.total_counter[node] = 0

        # DOES NOT MERGE TARGET COUNTER
        self.merge_counters()

        if self.current_target is None:
            self.current_target = self.select_target()

        destination = self.current_target
        packet_id = self.layercake.hostname + str(self.stats['packets_sent'])
        packet = {"ID": packet_id,
                  "dest": destination,
                  "source": self.layercake.hostname,
                  "route": [], "type": "DATA",
                  "time_stamp": Sim.now(),
                  "length": self.packet_length,
                  "data": self.test_packet_counter[destination]}
        self.sent_counter[destination] += 1
        self.test_packet_counter[destination] += 1

        if self.test_packet_counter[destination] % self.test_stream_length:
            # In Stream
            period = poisson(float(self.period * self.stream_period_ratio))
        else:
            # Finished Stream
            period = poisson(float(self.period))
            if DEBUG:
                self.logger.info("Finished Stream {} for {}, sleeping for {}".format(
                    self.test_packet_counter[destination] / self.test_stream_length,
                    destination,
                    period
                ))
            self.current_target = None
        return packet, period

    def packet_recv(self, packet):
        """

        :param packet:
        """
        self.merge_counters()
        self.received_counter[packet['source']] += 1
        self.result_packet_dl[packet['source']].append(packet['data'])

        if not (packet['data'] + 1) % self.test_stream_length:
            if DEBUG:
                self.logger.info("Got Stream {count} from {src} after {delay}".format(
                    count=(packet['data'] + 1) / self.test_stream_length,
                    src=packet['source'],
                    delay=Sim.now() - packet['time_stamp']
                ))
        del packet

    def dump_logs(self):
        """


        :return:
        """
        initial = super(CommsTrust, self).dump_logs()
        initial.update({
            'trust': self.trust_assessments,
            'trust_accessories': self.trust_accessories
        })
        return initial


class CommsTrustRoundRobin(CommsTrust):
    test_stream_length = 1

class SelfishCommsTrustRoundRobin(CommsTrustRoundRobin):
    def select_target(self):
        neighbours_by_distance = self.layercake.host.behaviour.get_nearest_neighbours()
        choices = [(v.name, 1.0/np.power(v.distance, 2)) for v in neighbours_by_distance]
        names, inv_sq_distances = zip(*choices)

        norm_distances = inv_sq_distances / sum(inv_sq_distances)
        new_target = np.random.choice(names, p=norm_distances)
        self.logger.warn("Selected {}".format(new_target))
        return new_target

    def activate(self):
        """
        Custom activation to set up fwd handler
        :return:
        """

        self.layercake.fwd_signal_hdlrs.append(self.query_fwd)

        super(SelfishCommsTrustRoundRobin, self).activate()

    def query_fwd(self, packet):
        """
        Only allow forwarding packets to neighbours
        :return:bool
        """
        if packet['dest'] == self.layercake.hostname or packet['source'] == self.layercake.hostname:
            drop_it = False
        else:
            fwd_pos = self.layercake.host.fleet.node_position_by_name(packet['dest'])
            neighbourly = self.layercake.net.is_neighbour(fwd_pos)
            if bool(neighbourly):
                drop_it = False

            else:
                self.logger.warn("Dropping Packet to {} as they're not a neighbour".format(packet["dest"]))
                if self.layercake.hostname=='n1' and DEBUG:
                    self.layercake.host.fleet.plot_axes_views().savefig('/dev/shm/test.png')
                drop_it = True

        return drop_it






class Null(Application):
    HAS_LAYERCAKE = False
    default_destination = None

    def generate_next_packet(self, period, destination, data=None, *args, **kwargs):
        """
        Does Nothing, says nothing
        :param period:
        :param destination:
        :param data:
        :param args:
        :param kwargs:
        """
        return None, 1

    @staticmethod
    def packet_recv(packet):
        """

        :param packet:
        """
        del packet
