# coding=utf-8
###########################################################################
# Copyright (C) 2007 by Justin Eskesen
# <jge@mit.edu>
#
# Copyright: See COPYING file that comes with this distribution
#
# This file is part of AUVNetSim, a library for simulating acoustic
# networks of fixed and mobile underwater nodes, written in Python.
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
# Contains all nodes' position - Only the sinks' position is needed
nodes_geo = {}
import collections
import math
import numpy as np
import operator

from aietes.Layercake.priodict import PriorityDictionary
from aietes.Tools import distance, broadcast_address, DEBUG


# DEBUG = True


def setup_routing(node, config):
    if config["Algorithm"] == "FBR":
        return FBR(node, config)
    elif config["Algorithm"] == "Static":
        return Static(node, config)
    else:
        return SimpleRoutingTable(node, config)


class SimpleRoutingTable(dict):
    def __init__(self, layercake, config):
        dict.__init__(self)
        self.layercake = layercake
        self.logger = layercake.logger.getChild(
            "{}".format(self.__class__.__name__))
        # if not DEBUG: self.logger.setLevel(logging.WARNING) # You always
        # forget about this
        self.has_routing_table = False
        self.packets = set([])

    def send_packet(self, packet):
        packet["level"] = 0
        packet["route"].append(
            (self.layercake.hostname, self.layercake.get_current_position()))
        try:
            packet["through"] = self[packet["dest"][0]]
            self.logger.warn("Routing {} through {} to {}".format(
                packet['ID'],
                packet['through'],
                packet['dest']
            ))
        except KeyError:
            packet["through"] = packet["dest"]
        except TypeError as e:
            raise RuntimeError("Possibly malformed/incomplete packet {}: raised {}".format(
                packet, e
            ))

        # Current hop position
        packet["source_position"] = self.layercake.get_current_position()
        try:
            packet["through_position"] = nodes_geo[packet["through"]]
        except KeyError:
            packet["through_position"] = None
        try:
            packet["dest_position"] = nodes_geo[packet["dest"]]
        except KeyError:
            packet["dest_position"] = None

        if DEBUG:
            self.logger.info("NET initiating TX of {}".format(packet['ID']))
        self.layercake.mac.initiate_transmission(packet)

    def on_packet_reception(self, packet):
        """
        If this is the final destination of the packet,
        pass it to the application layer
        otherwise, send it on.

        Note that packet deduplication is only conducted here
        :param packet: dict: received packet from MAC
        """

        if not self.have_duplicate_packet(packet):
            self.packets.add(packet["ID"])
            if packet["dest"] == self.layercake.hostname:
                self.logger.warn(
                    "received Packet from {src}".format(src=packet["source"]))
                self.layercake.recv(packet)
            elif packet["dest"] == broadcast_address:
                # Assume this has provided some info that the apps can get from
                # the routing layer in a bit.
                self.logger.debug("Heard Broadcast from {src}: storing record in route table".format(
                    src=packet["source"]))
                self[packet["source"]] = packet["route"]
            else:
                self.logger.info("Relaying packet from {src} to {dest}".format(
                    src=packet["source"], dest=packet['dest']))
                self.send_packet(packet)

    def need_explicit_ack(self, current_level, destination):
        return True

    def print_msg(self, msg):
        self.logger.info(msg)

    def have_duplicate_packet(self, packet):
        if packet["ID"] in self.packets:
            # This packet was already received, I should directly acknolwedge
            # it.
            self.logger.error(
                "Discarding a duplicated packet. ID: " + packet["ID"])
            return True

        return False


RoutingEntry = collections.namedtuple("RoutingEntry",
                                      ["next", "hops", "seq"])


class DSDV(SimpleRoutingTable):
    """
    Destination Sequenced Distance Vector protocol uses the Bellnman Ford Algo. to calculate paths based on hop-lengths

    Maintains two routing tables, one permemant and one advertised (i.e. stable and unstable)

    According to  Molins & Stojanovic (FAMA), reasonable to assume 'God' routing based on overheard DATA packets

    Cheat and add another field to the packets as they go down. This does not affect length.

    """
    has_routing_table = True  # NA
    # Time between full-table exchanges among nodes
    periodic_update_interval = 15
    # Enables Weighted Settling Time for updates before Advertisment
    wst_enabled = True
    settling_time = 6  # Minimum storage time before tx
    wst_factor = 0.875  # Fairly Obvious....
    buffer_enabled = True  # Buffer if no route available
    max_queue_len = 100  # Obvious
    max_queue_time = 30  # Obvious
    max_queued_per_dest = 5  # Q / Destination
    hold_times = 3  # how many periodic updates before purging a route
    route_agg_enabled = False  # Aggregated updates
    route_agg_time = 1  # seconds over which DSDV updates are aggregated

    def __init__(self, layercake, config=None):
        SimpleRoutingTable.__init__(self, layercake, config)
        # Sneaky trick to make 'recognisable' sequence numbers'
        self.seq_no = reduce(
            operator.xor, map(ord, self.layercake.hostname)) * 100

    def push_up(self, packet):
        """

        :param packet:
        """
        if hasattr(packet, 'table'):
            self.process_update_from_packet(packet)
        else:
            self.layercake.recv(packet)

    def send_packet(self, packet):
        """

        :param packet:
        """
        self.seq_no += 2
        tx_table = [RoutingEntry(self.layercake.hostname, 0, self.seq_no)]
        for dest, entry in self.items():
            assert isinstance(entry, RoutingEntry)
            tx_table.append(RoutingEntry(dest, entry.hops, entry.seq))

        packet['table'] = tx_table
        super(DSDV, self).send_packet(packet)

    def on_packet_reception(self, packet):
        """
        Process incoming tables before checking anything
        :param packet:
        :return:
        """
        self.process_update_from_packet(packet)
        super(DSDV, self).on_packet_reception(packet)

    def process_update_from_packet(self, packet):
        """
        Update the Routing Table based on incoming info from the packet
        :param packet:
        :return:
        """
        for dest, hop, seq_no in packet['table']:
            if dest not in self:
                new_entry = RoutingEntry(
                    packet['source'], hop + len(packet['route']), seq_no)
                self.logger.info(
                    "Creating entry for {}:{}".format(dest, new_entry))
                self[dest] = new_entry
            elif seq_no > self[dest].seq:
                new_entry = RoutingEntry(
                    packet['source'], hop + len(packet['route']), seq_no)
                self.logger.info(
                    "Updating entry for {}:{}".format(dest, new_entry))
                self[dest] = new_entry

        self.logger.info(
            "After update, table contains {}".format(self.items()))


class Static(SimpleRoutingTable):
    """ Note that it has not sense to use static routes when the network has mobile nodes. For MAC, CS-ALOHA or DACAP should be used.
        Variation 0: approximation to the transmission cone
        Variation 1: approximation to the receiver cone
        Variation 2: Dijkstras' algorithm for minimum power routes
    """

    def __init__(self, layercake, config):
        SimpleRoutingTable.__init__(self, layercake, config)
        self.nodes_rel_pos = {}
        # Contains already discovered positions
        self.nodes_pos = {layercake.hostname: layercake.get_current_position()}
        # Defines a cone where nodes may be of interest
        self.cone_angle = config["coneAngle"]
        # Defines a cone where nodes may be of interest
        self.cone_radius = config["coneRadius"]
        # Maximum distance between any two nodes in the network
        self.max_distance = config["maxDistance"]
        self.has_routing_table = False
        nodes_geo[layercake.hostname] = layercake.get_current_position()
        self.incoming_packet = None
        self.packets = set([])
        self.var = config["variation"]

    def on_packet_reception(self, packet):
        # if this is the final destination of the packet,
        # pass it to the application layer
        # otherwise, send it on...
        """

        :param packet:
        """
        if not self.have_duplicate_packet(packet):
            self.packets.add(packet["ID"])
            if packet["through"] == packet["dest"]:
                if packet["dest"] == self.layercake.hostname:
                    self.layercake.app.on_packet_reception(packet)
            else:
                self.send_packet(packet)

    def send_packet(self, packet):
        """

        :param packet:
        """
        self.incoming_packet = packet
        self.incoming_packet["route"].append(
            (self.layercake.hostname, self.layercake.get_current_position()))
        self.logger.debug("Processing packet with ID: " + packet["ID"])

        if self.incoming_packet["dest"] == "AnySink":
            self.incoming_packet["dest"] = self.closer_sink()

        if not self.has_routing_table:
            if self.var == 0:
                self.build_routing_table_0()
            elif self.var == 1:
                self.build_routing_table_1()
            elif self.var == 2:
                self.build_routing_table_2()

            self.has_routing_table = True

        self.incoming_packet["through"] = self[self.incoming_packet["dest"]]
        self.incoming_packet["through_position"] = nodes_geo[
            self.incoming_packet["through"]]

        # Current hop position
        self.incoming_packet[
            "source_position"] = self.layercake.get_current_position()
        self.incoming_packet["dest_position"] = nodes_geo[
            self.incoming_packet["dest"]]  # Final destination position

        self.incoming_packet["level"] = self.get_level_for(
            self.incoming_packet["through_position"])

        if self.incoming_packet["level"] is None:
            # This should actually never happen
            print self.incoming_packet

        self.layercake.mac.initiate_transmission(self.incoming_packet)

    def need_explicit_ack(self, current_level, destination):
        """

        :param current_level:
        :param destination:
        :return:
        """
        if self.layercake.hostname == destination:
            return True

        if not self.has_routing_table:
            if self.var == 0:
                self.build_routing_table_0()
            elif self.var == 1:
                self.build_routing_table_1()
            elif self.var == 2:
                self.build_routing_table_2()

            self.has_routing_table = True

        if self.get_level_for(nodes_geo[self[destination]]) < current_level:
            return True

        return False

    def closer_sink(self):
        """


        :return:
        """
        self.logger.debug("Looking for the closest Sink.")
        for name_item, pos_item in nodes_geo.iteritems():
            if name_item[0:4] == "Sink":
                self.nodes_rel_pos[name_item] = distance(
                    self.layercake.get_current_position(), pos_item)

        b = dict(
            map(lambda item: (item[1], item[0]), self.nodes_rel_pos.items()))
        min_key = b[min(b.keys())]
        return min_key

    def get_level_for(self, destination):
        """

        :param destination:
        :return:
        """
        levels = self.layercake.phy.level2distance

        for level, distance in levels.iteritems():
            if distance(self.layercake.get_current_position(), destination) <= distance:
                return level

    def build_routing_table_0(self):
        """ Transmission Cone approach.
        """
        self.logger.debug(
            "Building routing table for node " + self.layercake.hostname)
        self.nodes_rel_pos = {}
        for name_item, pos_item in nodes_geo.iteritems():
            if name_item == self.layercake.hostname:
                dist = 0
                angle = 0
                continue
            dist = distance(self.layercake.get_current_position(), pos_item)
            angle = self.layercake.get_current_position().anglewith(pos_item)

            self.nodes_rel_pos[name_item] = dist, angle

        for name_item, rel_pos_item in self.nodes_rel_pos.iteritems():

            next_hop_name = 'any'
            next_hop_rel_pos = self.max_distance, 0.0
            inside = False

            if rel_pos_item[0] < self.cone_radius:
                if DEBUG:
                    print 'Direct path is the best now'
                next_hop_name = name_item
            else:
                if DEBUG:
                    print 'When trying to reach node', name_item
                for i, j in self.nodes_rel_pos.iteritems():
                    if DEBUG:
                        print 'Trying a path through', i, 'with angle', j[1], 'and distance', j[0]
                    if j[1] < (rel_pos_item[1] + self.cone_angle / 2.0):
                        if j[1] > (rel_pos_item[1] - self.cone_angle / 2.0):
                            if inside:
                                if j[0] > self.cone_radius:
                                    if DEBUG:
                                        print 'case 1'
                                    continue

                                if j[0] > next_hop_rel_pos[0]:
                                    next_hop_name = i
                                    next_hop_rel_pos = j
                                    if DEBUG:
                                        print 'case 2'

                            else:
                                if j[0] < self.cone_radius:
                                    inside = True
                                    next_hop_name = i
                                    next_hop_rel_pos = j
                                    if DEBUG:
                                        print 'case 3'
                                elif j[0] < next_hop_rel_pos[0]:
                                    next_hop_name = i
                                    next_hop_rel_pos = j
                                    if DEBUG:
                                        print 'case 4'

            self[name_item] = next_hop_name

        self._update_routing_table_rec(dist, i)

    def build_routing_table_1(self):
        # Reception Cone
        """


        """
        self.logger.debug(
            "Building routing table for node " + self.layercake.hostname)
        self.nodes_rel_pos = {}

        for name_item, pos_item in nodes_geo.iteritems():
            if name_item == self.layercake.hostname:
                dist = 0.0
                tx_angle = 0.0
                rx_angle = 0.0
                continue

            dist = distance(self.layercake.get_current_position(), pos_item)
            tx_angle = self.layercake.get_current_position().anglewith(
                pos_item)
            rx_angle = pos_item.anglewith(
                self.layercake.get_current_position())

            self.nodes_rel_pos[name_item] = dist, tx_angle, rx_angle

        for name_item, rel_pos_item in self.nodes_rel_pos.iteritems():

            next_hop_name = 'any'
            next_hop_rel_pos = self.max_distance, 0.0, 0.0
            next_dist = self.max_distance
            next_angle = 90.0

            inside = False

            if rel_pos_item[0] < self.cone_radius:
                if DEBUG:
                    print 'Direct path is the best now'
                next_hop_name = name_item
            else:
                if DEBUG:
                    print 'When trying to reach node', name_item, rel_pos_item
                for i, j in self.nodes_rel_pos.iteritems():
                    if DEBUG:
                        print 'Trying a path through', i, 'with angle', j[1], 'and distance', j[0]

                    if j[1] < (rel_pos_item[1] + 90.0):
                        if j[1] > (rel_pos_item[1] - 90.0):
                            # This means that the node is going forward to the
                            # destination
                            if name_item == i:
                                recep_angle = rel_pos_item[2]
                            else:
                                recep_angle = nodes_geo[
                                    name_item].anglewith(nodes_geo[i])

                            if recep_angle < (rel_pos_item[2] + self.cone_angle / 2.0):
                                if recep_angle > (rel_pos_item[2] - self.cone_angle / 2.0):
                                    if inside:
                                        if j[0] > self.cone_radius:
                                            if DEBUG:
                                                print 'case 1'
                                            continue

                                        if j[0] > next_hop_rel_pos[0]:
                                            # if
                                            # nodes_geo[i].distanceto(nodes_geo[next_hop_name])
                                            # < next_dist:
                                            if abs(abs(recep_angle) - abs(rel_pos_item[2])) < abs(
                                                            abs(next_angle) - abs(rel_pos_item[2])):
                                                next_hop_name = i
                                                next_hop_rel_pos = j
                                                next_dist = dist(
                                                    nodes_geo[i], nodes_geo[next_hop_name])
                                                next_angle = recep_angle
                                                if DEBUG:
                                                    print 'case 2'

                                    else:
                                        if j[0] < self.cone_radius:
                                            inside = True
                                            next_hop_name = i
                                            next_hop_rel_pos = j
                                            next_dist = dist(
                                                nodes_geo[i], nodes_geo[next_hop_name])
                                            next_angle = recep_angle
                                            if DEBUG:
                                                print 'case 3'
                                        elif j[0] < next_hop_rel_pos[0]:
                                            next_hop_name = i
                                            next_hop_rel_pos = j
                                            next_dist = dist(
                                                nodes_geo[i], nodes_geo[next_hop_name])
                                            next_angle = recep_angle
                                            if DEBUG:
                                                print 'case 4'

            self[name_item] = next_hop_name

        self._update_routing_table_rec(dist, i)

    def _update_routing_table_rec(self, dist, i):
        # Apply recursivity
        for i in self:
            while self[i] != self[self[i]]:
                self[i] = self[self[i]]
                if DEBUG:
                    print i, self[i]

        # Check if the power level needed for each next hop allows us to
        # directly reaching destination, even being out of the cone
        for i in self:
            if self.get_level_for(nodes_geo[i]) is not None and self.get_level_for(nodes_geo[i]) <= self.get_level_for(
                    nodes_geo[self[i]]):
                self[i] = i
        if DEBUG:
            print i, nodes_geo[i], self[i], nodes_geo[self[i]], dist(self.layercake.get_current_position(),
                                                                     nodes_geo[self[i]])

    def build_routing_table_2(self):
        # Shortest path with level constraints
        """


        """
        self.logger.debug(
            "Building optimal routing table for node " + self.layercake.hostname)

        g = self.build_graph()
        d, p = self.dijkstra(g, self.layercake.hostname)

        for i, pos in nodes_geo.iteritems():
            end = i
            if end == self.layercake.hostname:
                continue
            path = []
            while 1:
                path.append(end)
                if end == self.layercake.hostname:
                    break
                end = p[end]
            path.reverse()
            self[i] = path[1]
            if DEBUG:
                print i, self[i]

    def build_graph(self):
        """


        :return:
        """
        graph = {}
        for fname, fpos in nodes_geo.iteritems():
            graph[fname] = {}
            for tname, tpos in nodes_geo.iteritems():
                if fname == tname:
                    continue
                d = distance(fpos, tpos)
                if self.get_level_ft(d) is not None:
                    # We can reach it using some power level
                    graph[fname][tname] = self.layercake.phy.distance2power[
                        self.get_level_ft(d)]

        return graph

    def get_level_ft(self, d):
        """

        :param d:
        :return:
        """
        levels = self.layercake.phy.level2distance
        for level, tx_range in levels.iteritems():
            if d <= tx_range:
                return level

    @staticmethod
    def dijkstra(g, start, end=None):
        """
        Find shortest paths from the start vertex to all
        vertices nearer than or equal to the end.

        The input graph g is assumed to have the following
        :param g:
        :param start:
        :param end:
        representation: A vertex can be any object that can
        be used as an index into a dictionary.  g is a
        dictionary, indexed by vertices.  For any vertex v,
        g[v] is itself a dictionary, indexed by the neighbors
        of v.  For any edge v->w, g[v][w] is the length of
        the edge.  This is related to the representation in
        <http://www.python.org/doc/essays/graphs.html>
        where Guido van Rossum suggests representing graphs
        as dictionaries mapping vertices to lists of neighbors,
        however dictionaries of edges have many advantages
        over lists: they can store extra information (here,
        the lengths), they support fast existence tests,
        and they allow easy modification of the graph by edge
        insertion and removal.  Such modifications are not
        needed here but are important in other graph algorithms.
        Since dictionaries obey iterator protocol, a graph
        represented as described here could be handed without
        modification to an algorithm using Guido's representation.

        Of course, g and g[v] need not be Python dict objects;
        they can be any other object that obeys dict protocol,
        for instance a wrapper in which vertices are URLs
        and a call to g[v] loads the web page and finds its links.

        The output is a pair (d,p) where d[v] is the distance
        from start to v and p[v] is the predecessor of v along
        the shortest path from s to v.

        dijkstra's algorithm is only guaranteed to work correctly
        when all edge lengths are positive. This code does not
        verify this property for all edges (only the edges seen
        before the end vertex is reached), but will correctly
        compute shortest paths even for some graphs with negative
        edges, and will raise an exception if it discovers that
        a negative edge has caused it to make a mistake.
        """
        d = {}  # dictionary of final distances
        p = {}  # dictionary of predecessors
        q = PriorityDictionary()  # est.dist. of non-final vert.
        q[start] = 0

        for v in q:
            d[v] = q[v]
            if v == end:
                break

            for w in g[v]:
                vwlength = d[v] + g[v][w]
                if w in d:
                    if vwlength < d[w]:
                        raise ValueError, \
                            "dijkstra: found better path to already-final vertex"
                elif w not in q or vwlength < q[w]:
                    q[w] = vwlength
                    p[w] = v
        return d, p

    def find_shortest_path(self, start, end):
        """
        Find a single shortest path from the given start vertex
        to the given end vertex.
        The input has the same conventions as dijkstra().
        The output is a list of the vertices in order along
        the shortest path.
        :param start:
        :param end:
        """
        g = self.build_graph()

        d, p = self.dijkstra(g, start, end)
        path = []
        while 1:
            path.append(end)
            if end == start:
                break
            end = p[end]
        path.reverse()
        return path


class FBR(SimpleRoutingTable):
    """ In this case, DACAP4FBR should be selected as MAC protocol.
        Variation 0: Transmission cone
        Variation 1: Reception cone (transmission cone with big apperture)
    """

    def __init__(self, layercake, config):
        SimpleRoutingTable.__init__(self, layercake, config)

        # Contains already discovered positions
        self.nodes_pos = {layercake.hostname: layercake.get_current_position()}
        nodes_geo[layercake.hostname] = layercake.get_current_position()
        self.cone_angle = config["coneAngle"]  # The cone aperture
        self.incoming_packet = None
        self.packets = set([])
        self.rx_cone = config["rx_cone"]

    def on_packet_reception(self, packet):
        """
        If this is the final destination of the packet,
        pass it to the application layer
        otherwise, send it on.

        Note that packet deduplication is only conducted here

        Also updates the global node_geo map

        Finally, implements monitor mode

        :param packet: obj: successfully received packet to go to layercake if for me
        :return: bool: was the packet for me?
        """
        global nodes_geo
        nodes_geo = self.layercake.host.fleet.node_map()
        if not self.have_duplicate_packet(packet):
            self.packets.add(packet["ID"])
            if packet["dest"] == self.layercake.hostname:
                self.layercake.recv(packet)
                return True
            elif self.layercake.query_drop_forward(packet=packet):
                self.logger.info("Dropping packet with ID: " + packet["ID"])
            else:
                self.logger.info("Forwarding packet with ID: " + packet["ID"])
                self.send_packet(packet)

                if self.layercake.monitor_mode:
                    self.layercake.monitor_mode(packet)

        else:
            self.logger.warn("Rejecting Dup {type} {ID} from {source} to {dest}".format(
                type=packet['type'],
                ID=packet.get('ID'),
                source=packet['source'],
                dest=packet['dest']
            ))
        return False

    def send_packet(self, packet):
        """
        Launches a packet into the MAC

        Also called by forwards
        :param packet:
        """
        self.incoming_packet = packet
        self.incoming_packet["route"].append(
            (self.layercake.hostname, self.layercake.get_real_current_position()))
        # Current hop position
        self.incoming_packet["source_position"] = self.layercake.get_current_position()

        if self.incoming_packet["dest"] == "AnySink":
            self.incoming_packet["dest"] = self.find_closer_sink()
            self.nodes_pos[self.incoming_packet["dest"]] = nodes_geo[
                self.incoming_packet["dest"]]  # This information is known (the sinks)
            self.incoming_packet["dest_position"] = self.nodes_pos[
                self.incoming_packet["dest"]]
        else:
            # Only Data packets come through this level, and they are already
            # directed to the sinks. I do know the position of the sinks.
            self.nodes_pos[self.incoming_packet["dest"]] = nodes_geo[
                self.incoming_packet["dest"]]
            self.incoming_packet["dest_position"] = self.nodes_pos[
                self.incoming_packet["dest"]]

        if self.is_neighbour(self.incoming_packet["dest_position"]):
            self.incoming_packet["through"] = self.incoming_packet["dest"]
            self.incoming_packet[
                "through_position"] = self.incoming_packet["dest_position"]
            self.incoming_packet["level"] = 0
        else:
            try:
                # If this works, then this is because I know the through
                self.incoming_packet["through"] = self[
                    self.incoming_packet["dest"]]
                self.incoming_packet["through_position"] = self.nodes_pos[
                    self.incoming_packet["through"]]
                self.incoming_packet["level"] = self.get_level_for(
                    self.incoming_packet["through_position"])
                # If I've been given a silly level, this will trip the exception.
                self.layercake.phy.level2delay(self.incoming_packet['level'])
            except KeyError:
                self.logger.debug("Route to {} not set. Starting Discovery".format(
                    self.incoming_packet['dest']
                ))
                self.incoming_packet["through"] = "ANY0"
                self.incoming_packet["through_position"] = 0
                self.incoming_packet["level"] = 0

        self.incoming_packet["level"] = self.layercake.query_pwr_adjust(packet)

        self.layercake.mac.initiate_transmission(self.incoming_packet)

    def is_reachable(self, current_level, dest_pos):
        """
        Is this level high enough to get to the destination?
        :param current_level: int
        :param dest_pos: ndarray([x,y,z])
        :return: bool
        """
        level_that_would_work = self.get_level_for(dest_pos)
        if level_that_would_work is None:
            self.logger.info("No Level will work to get to {}".format(dest_pos))
            reachable = False
        else:
            reachable = int(level_that_would_work) <= current_level

        return reachable

    def is_neighbour(self, dest_pos):
        """

        :param dest_pos:
        :return:
        """
        return self.is_reachable(0, dest_pos)

    def need_explicit_ack(self, current_level, destination):
        """

        :param current_level:
        :param destination:
        :return:
        """
        if self.layercake.hostname == destination:
            return True

        if destination in self:
            # When Broadcast flags go into the route table, the node_pos lookup dies miserably
            candidate = self[destination]
            if candidate not in self.nodes_pos:
                return True
            if self.get_level_for(self.nodes_pos[self[destination]]) is None:
                return True
            elif self.get_level_for(self.nodes_pos[self[destination]]) < current_level:
                return True
            else:
                self.logger.warn("Not ACKing to {}".format(destination))
                return False

        return True

    def get_level_for(self, destination):
        # Defaults to None for Reachablility assessment
        """

        :param destination:
        :return: int
        """
        new_level = None

        # ANY0/ANY1 etc
        if np.all(destination[0:3] == "ANY"):
            new_level = int(destination[3])
        else:
            r = distance(self.layercake.get_current_position(), destination)
            levels = zip(  # From the re-zipped
                *filter(  # unzipped filtered
                    lambda i: i[1] > r,  # list (l,d) where d > r
                    self.layercake.phy.level2distance.items()  # from available levels
                )
            )
            if levels:
                new_level = min(levels[0])  # Lowest Value Level

        return new_level

    def find_closer_sink(self):
        """


        :return:
        """
        self.logger.debug("Looking for the closest Sink.")
        self.nodes_rel_pos = {}
        for name_item, pos_item in nodes_geo.iteritems():
            self.nodes_rel_pos[name_item] = distance(
                self.layercake.get_current_position(), pos_item)

        b = dict(
            map(lambda item: (item[1], item[0]), self.nodes_rel_pos.items()))
        min_key = b[min(b.keys())]
        return min_key

    def i_am_a_valid_candidate(self, packet):
        """

        :param packet:
        :return:
        """
        valid = False
        if packet["dest"] == self.layercake.hostname:
            valid = True
        elif self.rx_cone == 0:
            """
            I will be a valid candidate if I am within the transmission cone.
            """
            source_pos = packet["source_position"]
            dest_pos = packet["dest_position"]

            if distance(self.layercake.get_current_position(), dest_pos) < distance(source_pos, dest_pos):
                a = distance(self.layercake.get_current_position(), dest_pos)
                b = distance(source_pos, dest_pos)
                c = distance(source_pos, self.layercake.get_current_position())

                if self._angular_d(a, b, c) > 0.99 or self._angular_d(a, b, c) < -0.99:
                    a = 0.0
                else:
                    a = math.degrees(
                        math.acos(self._angular_d(a, b, c)))

                valid = a <= self.cone_angle / 2.0

        elif self.rx_cone == 1:
            """
            I will be a valid candidate if I am within the reception cone.
            """
            source_pos = packet["source_position"]
            dest_pos = packet["dest_position"]
            valid = (distance(self.layercake.get_current_position(), dest_pos) < distance(source_pos, dest_pos))

        if valid:
            if self.layercake.query_drop_forward(packet):
                self.logger.debug("I'm a valid candidate for {} BUT IM IGNORING IT".format(
                    packet["dest"]
                ))
                valid = False
            else:
                self.logger.debug("I'm a valid candidate for {}".format(
                    packet["dest"]
                ))

        return valid

    def _angular_d(self, a, b, c):
        return (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)

    def add_node(self, name, pos):
        """

        :param name:
        :param pos:
        """
        self.nodes_pos[name] = pos

    def process_update_from_packet(self, dest, dest_pos, through, through_pos):
        """

        :param dest:
        :param dest_pos:
        :param through:
        :param through_pos:
        """
        self[dest] = through
        self.nodes_pos[dest] = dest_pos

    def have_duplicate_packet(self, packet):
        """

        :param packet:
        :return:
        """
        if packet["ID"] in self.packets:
            # This packet was already received, I should directly acknowledge
            # it
            self.logger.debug(
                "Discarding a duplicated packet. ID: " + packet["ID"])
            return True

        return False

    def select_route(self, candidates, current_through, attempts, destination):
        """

        :param candidates:
        :param current_through:
        :param attempts:
        :param destination:
        :return:
        """
        dist = {}
        ener = {}
        score = {}

        if not candidates:
            # There have been no answers
            self.logger.debug("Unable to reach " + current_through)

            if self.layercake.phy.collision_detected():
                self.logger.debug(
                    "There has been a collision, let's give it another chance!")
                return "2CHANCE", self.nodes_pos.get(current_through, 0)

            if current_through[0:3] != "ANY":
                # This is not a multicast RTS but a directed RTS which has been
                # not answered
                if attempts < 2:
                    self.logger.debug("Let's give it another chance.")
                    return "2CHANCE", 0
                else:
                    self.logger.debug("Starting multicast selection")
                    return "ANY0", 0

            if int(current_through[3]) != (len(self.layercake.phy.level2distance) - 1):
                # This is a multicast selection, but not with the maximum
                # transmission power level
                level = int(current_through[3]) + 1
                self.logger.debug("Increasing power to level {}".format(level))

                if self.is_reachable(level, nodes_geo[destination]):
                    self.logger.debug("{} is reachable".format(
                        destination
                    ))
                    return "NEIGH" + str(level), 0
                else:
                    self.logger.debug("{} is not reachable".format(
                        destination
                    ))
                    return "ANY" + str(level), 0
            else:
                self.logger.warn(
                    "Unable to reach any node within any transmission power. ABORTING transmission.")
                return "ABORT", 0

        else:
            # There have been answers: for a given transmission power, I should
            # always look for the one that is closer to the destination
            if destination in candidates:
                self.logger.debug("Selecting {} as direct route".format(
                    destination
                ))
                return destination, candidates[destination][2]

            # Now without energy criteria, I multiply by zero
            for name, de in candidates.iteritems():
                dist[name] = distance(de[2], nodes_geo[destination])
                ener[name] = de[1] * 0.0

            # Average score: min energy and min distance to the final
            # destination
            ee = dict(map(lambda item: (item[1], item[0]), ener.items()))
            max_ee = ee[max(ee.keys())]

            dd = dict(map(lambda item: (item[1], item[0]), dist.items()))
            max_dd = dd[max(dd.keys())]

            for name, de in candidates.iteritems():
                if ener[max_ee] > 0:
                    score[name] = dist[name] / dist[max_dd] + \
                                  ener[name] / ener[max_ee]
                else:
                    score[name] = dist[name] / dist[max_dd]

            sc = dict(map(lambda item: (item[1], item[0]), score.items()))
            min_score = sc[min(sc.keys())]
            self.logger.debug("Selecting {} as indirect route for {}".format(
                min_score,
                destination
            ))

            return min_score, self.nodes_pos[min_score]
