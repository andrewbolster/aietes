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

from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from numpy.random import poisson
from scipy.spatial.distance import squareform, pdist

from aietes.Tools import Sim, DEBUG, randomstr, broadcast_address, ConfigError

DEBUG = True


class Application(Sim.Process):
    """
    Generic Class for top level application layers
    """
    HAS_LAYERCAKE = True
    MONITOR_MODE = False
    random_delay = False

    def __init__(self, node, config, layercake):
        self._start_log(node)
        Sim.Process.__init__(self, name="{}({})".format(
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
        self.node = node
        self.layercake = layercake
        self.sim_duration = node.simulation.config.Simulation.sim_duration
        if self.HAS_LAYERCAKE:
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
                        "Taking Packet_Rate from config: {0!s}".format(self.packet_rate))
            elif packet_count > 0 and packet_rate == 0:
                # If packet count defined, only send our N packets
                self.packet_rate = packet_count / float(self.sim_duration)
                if DEBUG:
                    self.logger.debug(
                        "Taking Packet_Count from config: {0!s}".format(self.packet_rate))
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
            "App:{0!s}".format(self.__class__.__name__))

    def activate(self):

        if self.HAS_LAYERCAKE:
            self.layercake.tx_good_signal_hdlrs.append(self.signal_good_tx)
            self.layercake.tx_lost_signal_hdlrs.append(self.signal_lost_tx)
            self.layercake.app_rx_handler = self.packet_recv
            if self.MONITOR_MODE:
                self.layercake.activate(self.recv, monitor_mode=self.fwd)

        Sim.activate(self, self.lifecycle())

    def signal_good_tx(self, packetid):
        acked = False
        for sent in self.sent_log:
            if sent['ID'] == packetid:
                if DEBUG:
                    if sent['source'] is self.layercake.hostname:
                        self.logger.info("Confirmed TX of {} to {} at {} after {}".format(
                            sent['ID'], sent['dest'],
                            Sim.now(), Sim.now() - sent['time_stamp']
                        ))
                    else:
                        self.logger.info("Confirmed FWD for {} of {} to {} at {} after {}".format(
                            sent['source'], sent['ID'], sent['dest'],
                            Sim.now(), Sim.now() - sent['time_stamp']
                        ))
                sent['delivered'] = True
                sent['acknowledged'] = Sim.now()
                acked = True
        if not acked:
            self.logger.error("Have been told that a packet {} I can't remember sending has succeeded".format(
                packetid)
            )

    def signal_lost_tx(self, packetid):
        acked = False
        for sent in self.sent_log:
            if sent['ID'] == packetid:
                if sent['source'] is self.layercake.hostname:
                    self.logger.error("Failed TX of {} to {} at {} after {}".format(
                        sent['ID'], sent['dest'],
                        Sim.now(), Sim.now() - sent['time_stamp']
                    ))
                else:
                    self.logger.error("Failed FWD for {} of {} to {} at {} after {}".format(
                        sent['source'], sent['ID'], sent['dest'],
                        Sim.now(), Sim.now() - sent['time_stamp']
                    ))
                sent['delivered'] = False
                sent['acknowledged'] = Sim.now()
                acked = True
        if not acked:
            self.logger.error("Have been told that a packet {} I can't remember sending has succeeded".format(
                packetid)
            )

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
                        "Generated Payload {0!s}: Waiting {1!s}".format(packet['data'], period))
                self.layercake.send(packet)
                self.log_sent_packet(packet)

            yield Sim.hold, self, period

    def recv(self, from_below):
        """
        Called by RoutingTable on packet reception
        :param from_below:
        """
        if DEBUG:
            self.logger.info("Got Packet {id} from {src}".format(
                id=from_below['ID'],
                src=from_below['source']
            ))
        from_below['received'] = Sim.now()
        self.log_received_packet(from_below)
        self.packet_recv(from_below)

    def fwd(self, from_below):
        """
        Called by the routing layer on packet forwarded
        :param from_below:
        :return:
        """
        raise NotImplementedError("Shouldn't be in the base class!")

    def packet_recv(self, packet):
        raise NotImplementedError("Shouldn't be in the base class!")

    def log_sent_packet(self, packet):
        """

        :param packet:
        :return:
        """
        self.stats['packets_sent'] += 1
        packet['delivered'] = None
        self.sent_log.append(deepcopy(packet))

    def log_received_packet(self, packet):
        """
        Grab packet statistics
        :param packet:
        """
        source = packet["route"][0][0]

        if DEBUG:
            self.logger.info("App Packet received from {0!s}".format(source))
        self.stats['packets_received'] += 1

        delay = packet['delay'] = packet.get('received', Sim.now()) - packet['time_stamp']

        self.received_log.append(deepcopy(packet))

        # Ignore first hop (source)
        hops = len(packet['route'])

        self.stats['packets_time'] += delay
        self.stats['packets_hops'] += hops
        self.stats['packets_dhops'] += (delay / hops)

        self.logger.debug("Packet received from {0!s} over {1:d} hops with a delay of {2!s} (d/h={3!s})".format(
            source, hops, str(delay), str(delay / hops)))

    def dump_stats(self):
        """
        Calculated Throughput/Load based on (Total Bits In/Out / s) * hops
        Avg Lengths and Delays in the expected way (defaulting to 0,np.inf if no packets recieved)


        :return: dict: Application Stats including any stats from the physical layer
        :raise RuntimeError:
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
                "throughput": throughput,  # bps
                "offeredload": offeredload,  # bps
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
        ], "type": "DATA", "time_stamp": Sim.now(), "length": self.layercake.packet_length}
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

    def activate(self):
        self.forced_nodes = self.layercake.host.fleet.node_names()
        if self.forced_nodes:
            for node in self.forced_nodes:
                if node != self.layercake.hostname:
                    self.total_counter[node] = 0
        self.test_packet_counter = Counter(self.total_counter)
        self.result_packet_dl = {name: []
                                 for name in self.total_counter.keys()}
        super(RoutingTest, self).activate()

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
        if indirect_nodes:
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
                  "length": self.layercake.packet_length,
                  "data": None}
        period = poisson(float(period))
        return packet, period

    def packet_recv(self, packet):
        """

        :param packet:
        """
        self.merge_counters()
        self.received_counter[packet['source']] += 1

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


class Trust(RoutingTest):
    """
    Base class for Trust assessments

    Default Trust Assessment Period of 10 minutes
    Default to Random Delay < 60 s before first cycle - reduces contention.

    Required Child Method Invocations:
        Must set one or more method or functions as `tick_assessors`, which
        should return a node-indexed Series of measurement Series.
        These will be invoked at each 'tick', and combined into a

    Optional Child Methods:
        `update_accessories`()

    """
    trust_assessment_period = 600

    def __init__(self, *args, **kwargs):
        super(Trust, self).__init__(*args, **kwargs)
        # List of methods called per tick.
        self.tick_assessors = []
        # List of methods called per packet received
        self.packet_receivers = []
        # While it's tempting to preallocate this as a DataFrame, it is over 10 times slower than dict ops
        self.trust_assessments = {}  # My generated trust metrics, [node][t][observation Series]
        # Extra information that might be interesting in the longer term.
        self.trust_accessories = {}
        self.last_trust_assessment = None

    def activate(self):
        """
        Called By Node Activation to launch internal timers and configs
        """
        super(Trust, self).activate()

    def update_accessories(self):
        """
        Add other useful information from this time period to the accessory store
        :return:
        """
        pass

    def packet_recv(self, packet):
        """
        Multi Function Aware Packet Handler
        These functions should not expect any interdependence in packet content as if they're getting it individually
        :param packet:
        :return:
        """
        for receiver in self.packet_receivers:
            receiver(deepcopy(packet))

    def tick(self):
        """
        Processing performed periodically depending on the simulation time

        Should populate self.trust_assessments as a node-array dict of metric dicts and optionally
        self.trust_accessories as a metric-array dict.

        EG: for a node "nodename" that is first observed in int third trust period
        trust_assessments = {
            'nodename': [
                Series([]),
                Series([]),
                Series([]),
                Series({'metrica':valuea, 'metricb': valueb,...,})
                Series({'metrica':valuea, 'metricb': valueb,...,})


            ]
        }

        """
        if not Sim.now() % self.trust_assessment_period and Sim.now() > 0:
            trust_period = Sim.now() // self.trust_assessment_period

            tick_assessments = [tick() for tick in self.tick_assessors]
            assert all([not isinstance(t, list) for t in
                        tick_assessments]), "All Tick Assessors should return a 1-d list of values"
            tick_keys = { i for t in tick_assessments for i in t.keys().tolist() }
            target_stats = {
                node:
                    pd.concat([pd.Series(tick[node]) for tick in tick_assessments if node in tick])
                for node in tick_keys
                }
            for node, per_target_stats in target_stats.iteritems():
                if node not in self.trust_assessments:
                    self.trust_assessments[node] = [pd.Series([]) for _ in range(trust_period)]
                self.trust_assessments[node].append(per_target_stats)

            self.last_accessed_rx_packet = len(self.received_log) - 1

            self.update_accessories()

    def dump_logs(self):
        """


        :return:
        """
        initial = super(Trust, self).dump_logs()
        initial.update({
            'trust': self.trust_assessments,
            'trust_accessories': self.trust_accessories
        })
        return initial


class BehaviourTrust(Trust):
    """
    Reimplementation of Physical Trust Analysis work.

    This application maintains it's own separate environmental record based on packet reception.

    It assumes that each packet received contains an updated global position log.
    #TODO this needs to be extended to:
        Take snapshot from time of transmission instead
        Actually manage a proper map

    Performs the same Routing Test behaviour as Comms Trust

    """

    def __init__(self, *args, **kwargs):
        super(BehaviourTrust, self).__init__(*args, **kwargs)

        self.pos_log = []
        self._use_fleet_log = True

        self.tick_assessors.extend([self.physical_metrics])
        self.packet_receivers.extend([self.update_log_from_packet])

    def physical_metrics(self):
        """
        This should really be rewritten (along with bounos.Metrics) to pull out these
        metrics rather than having them coupled to DataPackage

        Assume that we only have a pos_log
        """

        if self._use_fleet_log:
            # Assume magic
            full_pos_log = self.node.fleet.node_poslogs()
            last_relevant_time = Sim.now() - self.trust_assessment_period
            pos_log = full_pos_log[:, :, last_relevant_time:]
        else:
            # Assume less magic: that we get position info from every packet received
            full_pos_log = np.asarray(self.pos_log)
            if full_pos_log:
                pos_log = full_pos_log[:, :, self.last_accessed_rx_packet:]
            else:
                pos_log = None

        return BehaviourTrust.physical_metrics_from_position_log(pos_log, self.node.fleet.node_names(),
                                                                 single_value=True)

    @classmethod
    def physical_metrics_from_position_log(cls, pos_log, names=None, single_value=True):
        """
        Assumes [n,3,t] ndarray of positions

        Perform Local-versions of
                Metrics.DeviationOfHeading
                PernodeSpeed
                PernodeInternodeDistanceAvg

        INHD -
            $ m_i^t = h_{avg}^t -h_i $

        Speed -
            $ m_i^t = |h_i^t| $

        INDD -
            $ m_i^t =
            :param pos_log:
            :param names:
            :param single_value:

        """

        phys_stats = pd.Series([])

        if pos_log is None:
            # Can't assess trust, pass through empty stats, or half-generate stats series if names are available
            if names is not None:
                for i, node in enumerate(names):
                    phys_stats[node] = pd.Series({
                        "INDD": None,
                        "Speed": None,
                        "INHD": None
                    })

        else:
            n, d, t = pos_log.shape
            assert d == 3, d

            if names is None:
                names = [str(i) for i in range(n)]

            # Diff assumes the last axis (time) is the differential (i.e. dt, which is what we want)
            vec_log = np.diff(pos_log)  # (n,3,t-1)

            # Deviation of Heading (INHD)
            # v = np.asarray([data.deviation_from_at(data.average_heading(time), time) for time in range(int(data.tmax))])
            # c = np.average(v, axis=1)
            avg_vec = np.average(vec_log, axis=0)
            assert avg_vec.shape == (3, t - 1), avg_vec.shape
            inhd = np.linalg.norm(avg_vec - vec_log, axis=1)
            assert inhd.shape == (n, t - 1), inhd.shape

            # Per Node Speed
            # v = np.asarray([map(mag, data.heading_slice(time)) for time in range(int(data.tmax))])
            # c = np.average(v, axis=1)
            mag_log = np.linalg.norm(vec_log, axis=1)  # (n,t)

            # Per Node Internode Distance Deviation (INDD)
            # Deviation from me to the fleet centroid compared to the average inter node distance
            # v = np.asarray([data.distances_from_average_at(time) for time in range(int(data.tmax))])
            # c = np.asarray([data.inter_distance_average(time) for time in range(int(data.tmax))])
            inter_node_distance_matrices = np.asarray([squareform(pdist(_pt)) for _pt in np.rollaxis(pos_log, 2)]).T
            avg_pos = np.average(pos_log, axis=0)
            avg_dist = np.linalg.norm(avg_pos - pos_log, axis=1)
            indd = np.average(inter_node_distance_matrices, axis=1)
            assert indd.shape == (n, t), indd.shape

            for i, node in enumerate(names):
                phys_stats[node] = pd.Series({
                    "INDD": indd[i],
                    "Speed": mag_log[i],
                    "INHD": inhd[i]
                })
                if single_value:
                    phys_stats[node] = phys_stats[node].apply(np.mean)

        return phys_stats

    def update_log_from_packet(self, packet):
        """
        Application Level processing of packet. Must call super class
        :param packet:
        """
        self.pos_log.append(self.node.fleet.node_positions())


class CommsTrust(Trust):
    """
    Vaguely Emulated Bellas Traffic Scenario
    Trust Values retained per interval:
        Packet Loss Rate

    Default Trust Assessment Period of 10 minutes
    Default to Random Delay < 60 s before first cycle - reduces contension.

    #TODO Extend this to perform Grey Factor Analysis in real time?

    """
    test_stream_length = 6
    stream_period_ratio = 0.1

    random_delay = 60
    metrics_string = "ATXP,ARXP,ADelay,ALength,Throughput,PLR"
    MONITOR_MODE = True

    def __init__(self, *args, **kwargs):
        super(CommsTrust, self).__init__(*args, **kwargs)

        # Extra information that might be interesting in the longer term.
        self.trust_accessories.update(
            {'queue_length': [],
             'collisions': []
             })
        self.tick_assessors.extend([self.rx_trust_metrics, self.tx_trust_metrics])
        self.packet_receivers.extend([self.update_counters])

        self.current_target = None
        self.forced_nodes = []
        self.test_packet_counter = Counter()
        self.result_packet_dl = {}
        self.use_median = self.config['median']

    def activate(self):
        """
        Called By Node Activation to launch internal timers and configs

        """
        self.last_accessed_rx_packet = None

        super(CommsTrust, self).activate()

    @classmethod
    def get_metrics_from_received_packet(cls, packet):
        """
        Extracts the following measurements from a packet
            - tx power (TXP)
            - rx power (RXP)
            - delay (Delay)
            - data length (Length)
        :param packet: dict: received packet
        :return:pd.Series : keys=[TXP, RXP, Delay]
        :raises KeyError
        """
        pkt_indexes_map = {
            'tx_pwr_db': "TXP",
            'rx_pwr_db': "RXP",
            'delay': "Delay"
        }
        return pd.Series({v: packet[k] for k, v in pkt_indexes_map.items()})

    def get_metrics_from_batch(self, batch):
        """
        Extracts per-packet metrics, [ TXP, RXP, Delay, Length], averages them, and tags on the following cross packet metrics
            - rx-throughput
        :param batch:
        :return: pd.Series:
        """

        nodepktstats = []
        throughput = 0  # SUM Length

        for pkt in batch:
            nodepktstats.append(self.get_metrics_from_received_packet(pkt))
            throughput += pkt['length']

        if throughput:
            try:
                # Bring individual observations into a frame
                df = pd.concat(nodepktstats, axis=1)
                if self.use_median:
                    s = df.median(axis=1)
                else:
                    s = df.mean(axis=1)
                # Prepend the keys with "A" to denote average values
                s.index = ["A" + k for k in s.keys()]
                # Append the Throughput to the series and return
                s['RXThroughput'] = throughput
                return s
            except:
                self.logger.exception("PKTS:{},TP:{},NANMEAN:{}".format(
                    nodepktstats, throughput, np.nanmean(nodepktstats, axis=0)
                ))
                raise
        else:
            return pd.Series([])

    def update_accessories(self):
        self.trust_accessories['queue_length'].append(len(self.layercake.mac.outgoing_packet_queue))
        self.trust_accessories['collisions'].append(len(self.layercake.phy.transducer.collisions))

    def rx_trust_metrics(self):

        relevant_rx_packets = self.received_log[self.last_accessed_rx_packet:]

        rx_stats = pd.Series([])
        pkt_batches = {}

        for pkt in relevant_rx_packets:
            # FIXME this should maybe be route[-1]
            try:
                pkt_batches[pkt['source']].append(pkt)
            except KeyError:
                pkt_batches[pkt['source']] = [pkt]

        for node, pkt_batch in pkt_batches.iteritems():
            # Take the Mean and total throughput for all messages sent by node
            rx_stats[node] = self.get_metrics_from_batch(pkt_batch)
            # avg(tx pwr, rx pwr, delay, length), rx throughput
        return rx_stats

    def tx_trust_metrics(self):
        last_relevant_time = Sim.now() - self.trust_assessment_period
        relevant_acked_packets = []

        for i, p in enumerate(self.sent_log):
            # Only concern yourself with packets actually sent out (i.e. made it beyond the queue)
            if p.get('acknowledged', -1) > last_relevant_time:
                relevant_acked_packets.append(p)

        tx_stats = pd.Series([])
        for node in self.total_counter.keys():
            # Throughput is the total length of data transmitted this frame to this node
            tx_throughput = map(
                lambda p: float(p['length']),
                filter(
                    lambda p: node in [p['dest'], p['through']],
                    relevant_acked_packets)
            )
            if tx_throughput:
                tx_throughput = np.sum(tx_throughput)
            else:
                tx_throughput = 0.0

            # PLR is the average amount of packets we are sure have been lost in the last time frame
            # Including pkts that have been acked since last time.
            plr = map(
                lambda p: float(not p['delivered']),
                filter(
                    # We know the fate of all acked packets
                    lambda p: p['dest'] == node,
                    relevant_acked_packets)
            )
            if plr and tx_throughput > 0.0:
                plr = np.nanmean(plr)
            else:
                plr = 0.0

            tx_stats[node] = pd.Series({
                'PLR': plr if not np.isnan(plr) else 0.0,
                'TXThroughput': tx_throughput
            })
        return tx_stats

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
                  "length": self.layercake.packet_length,
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

    def fwd(self, packet):
        """
        Interception point for monitor-mode forwards, i.e. packets I'm sending but I didn't
        initiate.

        Forwarded Packets represent both a packet received and a packet sent; the callbacks
        and statistics for these are handled separately (i.e. in RX we're concerned with the
        power, delay, etc., in TX we're concerned about the ack callback)

        :param packet:
        :return:
        """
        self.log_sent_packet(deepcopy(packet))
        self.log_received_packet(deepcopy(packet))

    def update_counters(self, packet):
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


class CommsTrustRoundRobin(CommsTrust):
    test_stream_length = 1


class CombinedTrust(BehaviourTrust, CommsTrustRoundRobin):
    """
    Fusion Class of Comms and Behaviour with some deconflicting functionality.
    In theory
    """

    def __init__(self, *args, **kwargs):
        super(CombinedTrust, self).__init__(*args, **kwargs)


class SelfishTargetSelection(CommsTrustRoundRobin):
    def select_target(self):
        neighbours_by_distance = self.layercake.host.behaviour.get_nearest_neighbours()
        choices = [(v.name, 1.0 / np.power(v.distance, 2)) for v in neighbours_by_distance]
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

        super(SelfishTargetSelection, self).activate()

    def query_fwd(self, packet):
        """
        Only allow forwarding packets to neighbours
        :param packet:
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
                if self.layercake.hostname == 'n1' and DEBUG:
                    self.layercake.host.fleet.plot_axes_views().savefig('/dev/shm/test.png')
                drop_it = True

        return drop_it


class CombinedSelfishTargetSelection(BehaviourTrust, SelfishTargetSelection):
    """
    Fusion Class of Selfish and Behaviour Trust with some deconflicting functionality.
    In theory
    """

    def __init__(self, *args, **kwargs):
        super(CombinedSelfishTargetSelection, self).__init__(*args, **kwargs)


class BadMouthingPowerControl(CommsTrustRoundRobin):
    """
    INCREASES the power to everyone except for the given bad_mouth target
    """
    bad_mouth_target = 'n0'

    def activate(self):
        """
        Custom activation to set up fwd handler
        :return:
        """

        self.layercake.pwd_signal_hdlr = self.query_pwr

        super(BadMouthingPowerControl, self).activate()

    def query_pwr(self, packet):
        """
        Increase the power to everyone except the target
        Called by send_packet in RoutingLayer via Layercake
        :param packet:
        :return:bool
        """
        if packet['dest'] != self.bad_mouth_target:
            try:
                new_level = str(max(0, min(int(packet['level']) + 1, int(self.layercake.phy.max_level))))
            except:
                self.logger.error("Something is very very wrong with {}".format(
                    packet
                ))
                raise
            self.logger.warn("Adjusted {}->{} from {} to {}".format(
                packet['source'], packet['dest'],
                packet['level'], new_level
            ))
        else:
            new_level = packet["level"]

        return new_level


class CombinedBadMouthingPowerControl(BehaviourTrust, BadMouthingPowerControl):
    """
    Fusion Class of Malicious and Behaviour trust with some deconflicting functionality.
    In theory
    """

    def __init__(self, *args, **kwargs):
        super(CombinedBadMouthingPowerControl, self).__init__(*args, **kwargs)


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
