###########################################################################
#    Copyright (C) 2007 by Justin Eskesen                                      
#    <jge@mit.edu>                                                             
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
nodes_geo = {} # Contains all nodes' position - Only the sinks' position is needed
import math
from priodict import priorityDictionary

from aietes.Tools import distance, broadcast_address, debug, logging, Sim
import collections, operator

debug = False
def SetupRouting(node, config):
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
        self.logger = layercake.logger.getChild("{}".format(self.__class__.__name__))
        #if not debug: self.logger.setLevel(logging.WARNING) # You always forget about this
        self.has_routing_table = False
        self.packets = set([])

    def SendPacket(self, packet):
        packet["level"]=0
        packet["route"].append((self.layercake.hostname, self.layercake.get_current_position()))
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
                packet,e
            ))


        packet["source_position"] = self.layercake.get_current_position() #Current hop position
        try:
            packet["through_position"] = nodes_geo[packet["through"]]
        except KeyError:
            packet["through_position"] = None
        try:
            packet["dest_position"] = nodes_geo[packet["dest"]]
        except KeyError:
            packet["dest_position"] = None

        if debug: self.logger.info("NET initiating TX of {}".format(packet['ID']))
        self.layercake.mac.InitiateTransmission(packet)

    def OnPacketReception(self, packet):
        # If this is the final destination of the packet,
        # pass it to the application layer
        # otherwise, send it on.

        if not self.IsDuplicated(packet):
            self.packets.add(packet["ID"])
            if packet["dest"] == self.layercake.hostname:
                self.logger.debug("received Packet from {src}".format(src=packet["source"]))
                self.layercake.recv(packet)
            elif packet["dest"] == broadcast_address:
                # Assume this has provided some info that the apps can get from the routing layer in a bit.
                self.logger.debug("Heard Broadcast from {src}: storing record in route table".format(src=packet["source"]))
                self[packet["source"]]=packet["route"]
            else:
                self.logger.info("Relaying packet from {src} to {dest}".format(src=packet["source"], dest=packet['dest']))
                self.SendPacket(packet)

    
    def NeedExplicitACK(self, current_level, destination):
        return True

    def PrintMessage(self, msg):
        self.logger.info(msg)

    def IsDuplicated(self, packet):
        if packet["ID"] in self.packets:
            # This packet was already received, I should directly acknolwedge it.
            self.logger.debug("Discarding a duplicated packet. ID: "+packet["ID"])
            return True

        return False

RoutingEntry = collections.namedtuple("RoutingEntry",
                                      ["next","hops","seq"])

class DSDV(SimpleRoutingTable):
    """
    Destination Sequenced Discance Vector protocol uses the Bellnman Ford Algo. to calculate paths based on hop-lengths

    Maintains two routing tables, one permemant and one advertised (i.e. stable and unstable)

    According to  Molins & Stojanovic (FAMA), reasonable to assume 'God' routing based on overheard DATA packets

    Cheat and add another field to the packets as they go down. This does not affect length.

    """
    has_routing_table = True        # NA
    periodic_update_interval = 15   # Time between full-table exchanges among nodes
    wst_enabled = True              # Enables Weighted Settling Time for updates before Advertisment
    settling_time = 6               # Minimum storage time before tx
    wst_factor = 0.875              # Fairly Obvious....
    buffer_enabled = True           # Buffer if no route available
    max_queue_len = 100             # Obvious
    max_queue_time = 30             # Obvious
    max_queued_per_dest = 5         # Q / Destination
    hold_times = 3                  # how many periodic updates before purging a route
    route_agg_enabled = False       # Aggregated updates
    route_agg_time = 1              # seconds over which DSDV updates are aggregated


    def __init__(self, layercake, config=None):
        SimpleRoutingTable.__init__(self,layercake,config)
        self.seq_no = reduce(operator.xor,map(ord,self.layercake.hostname))*100                  #Sneaky trick to make 'recognisable' sequence numbers'

    def push_up(self,packet):
        if hasattr(packet, 'table'):
            self.process_update_from_packet(packet)
        else:
            self.layercake.recv(packet)

    def SendPacket(self, packet):

        self.seq_no+=2
        tx_table = [RoutingEntry(self.layercake.hostname, 0, self.seq_no)]
        for dest,entry in self.items():
            tx_table.append((dest,entry.hops,entry.seq))

        packet['table']=tx_table
        super(DSDV,self).SendPacket(packet)

    def OnPacketReception(self, packet):
        """
        Process incoming tables before checking anything
        :param packet:
        :return:
        """
        self.process_update_from_packet(packet)
        super(DSDV,self).OnPacketReception(packet)

    def process_update_from_packet(self, packet):
        """
        Update the Routing Table based on incoming info from the packet
        :param packet:
        :return:
        """
        for dest, hop, seq_no in packet['table']:
            if not self.has_key(dest):
                new_entry=RoutingEntry(packet['source'],hop+len(packet['route']),seq_no)
                self.logger.info("Creating entry for {}:{}".format(dest,new_entry))
                self[dest]=new_entry
            elif seq_no > self[dest].seq:
                new_entry=RoutingEntry(packet['source'],hop+len(packet['route']),seq_no)
                self.logger.info("Updating entry for {}:{}".format(dest,new_entry))
                self[dest]=new_entry

        self.logger.info("After update, table contains {}".format(self.items()))

class Static(SimpleRoutingTable):
    ''' Note that it has not sense to use static routes when the network has mobile nodes. For MAC, CS-ALOHA or DACAP should be used.
        Variation 0: approximation to the transmission cone
        Variation 1: approximation to the receiver cone
        Variation 2: Dijkstras' algorithm for minimum power routes
    '''

    def __init__(self, layercake, config):
        SimpleRoutingTable.__init__(self, layercake, config)
        self.nodes_pos = {layercake.hostname:layercake.get_current_position()} # Contains already discovered positions
        self.cone_angle = config["coneAngle"] # Defines a cone where nodes may be of interest
        self.cone_radius = config["coneRadius"] # Defines a cone where nodes may be of interest
        self.max_distance = config["maxDistance"] # Maximum distance between any two nodes in the network
        self.has_routing_table = False
        nodes_geo[layercake.hostname] = layercake.get_current_position()
        self.incoming_packet = None
        self.packets = set([])
        self.var = config["variation"]
    

    def OnPacketReception(self, packet):
        # if this is the final destination of the packet,
        # pass it to the application layer
        # otherwise, send it on...
        if not self.IsDuplicated(packet):
            self.packets.add(packet["ID"])
            if packet["through"] == packet["dest"]:
                if packet["dest"] == self.layercake.hostname:
                    self.layercake.app.OnPacketReception(packet)
            else:
                self.SendPacket(packet)


    def SendPacket(self, packet):
        self.incoming_packet = packet
        self.incoming_packet["route"].append((self.layercake.hostname, self.layercake.get_current_position()))
        self.logger.debug("Processing packet with ID: "+packet["ID"])

        if self.incoming_packet["dest"]=="AnySink":
            self.incoming_packet["dest"]=self.CloserSink()
        
        if self.has_routing_table == False:
            if self.var==0:
                self.BuildRoutingTable0()
            elif self.var==1:
                self.BuildRoutingTable1()
            elif self.var==2:
                self.BuildRoutingTable2()

            self.has_routing_table = True

        self.incoming_packet["through"] = self[self.incoming_packet["dest"]]
        self.incoming_packet["through_position"] = nodes_geo[self.incoming_packet["through"]]

        self.incoming_packet["source_position"] = self.layercake.get_current_position() #Current hop position
        self.incoming_packet["dest_position"] = nodes_geo[self.incoming_packet["dest"]] #Final destination position

        self.incoming_packet["level"] = self.GetLevel(self.incoming_packet["through_position"])

        if self.incoming_packet["level"] == None:
            # This should actually never happen
            print self.incoming_packet
        
        self.layercake.mac.InitiateTransmission(self.incoming_packet)
        

    def NeedExplicitACK(self, current_level, destination):
        if self.layercake.hostname == destination:
            return True

        if self.has_routing_table == False:
            if self.var==0:
                self.BuildRoutingTable0()
            elif self.var==1:
                self.BuildRoutingTable1()
            elif self.var==2:
                self.BuildRoutingTable2()

            self.has_routing_table = True

        if self.GetLevel(nodes_geo[self[destination]]) < current_level:
            return True
        
        return False


    def CloserSink(self):
        self.logger.debug("Looking for the closest Sink.")
        self.nodes_rel_pos = {}        
        for name_item, pos_item in nodes_geo.iteritems():
            if name_item[0:4] == "Sink":
                self.nodes_rel_pos[name_item] = distance(self.layercake.get_current_position(),pos_item)

        b = dict(map(lambda item: (item[1],item[0]),self.nodes_rel_pos.items()))
        min_key = b[min(b.keys())]        
        return min_key     


    def GetLevel(self, destination):
        levels = self.layercake.phy.level2distance

        for level, distance in levels.iteritems():
            if distance(self.layercake.get_current_position(),destination)<=distance:
                return level

    def Update(self, dest, dest_pos, through, through_pos):
        pass
    

    def BuildRoutingTable0(self):
        ''' Transmission Cone approach.
        '''
        self.logger.debug("Building routing table for node "+self.layercake.hostname)
        self.nodes_rel_pos = {}
        for name_item, pos_item in nodes_geo.iteritems():
            if name_item == self.layercake.hostname:
                distance = 0
                angle = 0
                continue
            distance = distance(self.layercake.get_current_position(),pos_item)
            angle = self.layercake.get_current_position().anglewith(pos_item)

            self.nodes_rel_pos[name_item] = distance, angle

        for name_item, rel_pos_item in self.nodes_rel_pos.iteritems():

            next_hop_name = 'any'
            next_hop_rel_pos = self.max_distance, 0.0
            inside = False

            if rel_pos_item[0] < self.cone_radius:
                if debug:
                    print 'Direct path is the best now'
                next_hop_name = name_item
            else:
                if debug:
                    print 'When trying to reach node', name_item
                for i, j in self.nodes_rel_pos.iteritems():
                    if debug:
                        print 'Trying a path through', i, 'with angle', j[1], 'and distance', j[0]
                    if j[1] < ( (rel_pos_item[1]+self.cone_angle/2.0) ):
                        if j[1] > ( (rel_pos_item[1]-self.cone_angle/2.0) ):
                            if inside:
                                if j[0] > self.cone_radius:
                                    if debug:
                                        print 'case 1'
                                    continue

                                if j[0] > next_hop_rel_pos[0]:
                                    next_hop_name = i
                                    next_hop_rel_pos = j
                                    if debug:
                                        print 'case 2'                                

                            else:
                                if j[0] < self.cone_radius:
                                    inside = True
                                    next_hop_name = i
                                    next_hop_rel_pos = j
                                    if debug:
                                        print 'case 3'
                                elif j[0] < next_hop_rel_pos[0]:
                                    next_hop_name = i
                                    next_hop_rel_pos = j
                                    if debug:
                                        print 'case 4'

            self[name_item] = next_hop_name

        # Apply recursivity
        for i in self:
            while self[i] != self[self[i]]:
                self[i] = self[self[i]]
                if debug:
                    print i, self[i]

        # Check if the power level needed for each next hop allows us to directly reaching destination, even being out of the cone
        for i in self:
            if self.GetLevel(nodes_geo[i])!=None and self.GetLevel(nodes_geo[i])<=self.GetLevel(nodes_geo[self[i]]):
                self[i]=i

        if debug:
            print i, nodes_geo[i], self[i], nodes_geo[self[i]], distance(self.layercake.get_current_position(),nodes_geo[self[i]])
            

    def BuildRoutingTable1(self):
        # Reception Cone
        self.logger.debug("Building routing table for node "+self.layercake.hostname)
        self.nodes_rel_pos = {}
        
        
        for name_item, pos_item in nodes_geo.iteritems():
            if name_item == self.layercake.hostname:
                distance = 0.0
                tx_angle = 0.0
                rx_angle = 0.0
                continue

            distance = distance(self.layercake.get_current_position(),pos_item)
            tx_angle = self.layercake.get_current_position().anglewith(pos_item)
            rx_angle = pos_item.anglewith(self.layercake.get_current_position())

            self.nodes_rel_pos[name_item] = distance, tx_angle, rx_angle

        for name_item, rel_pos_item in self.nodes_rel_pos.iteritems():

            next_hop_name = 'any'
            next_hop_rel_pos = self.max_distance, 0.0, 0.0
            next_dist = self.max_distance
            next_angle = 90.0
            
            inside = False

            if rel_pos_item[0] < self.cone_radius:
                if debug:
                    print 'Direct path is the best now'
                next_hop_name = name_item
            else:
                if debug:
                    print 'When trying to reach node', name_item, rel_pos_item
                for i, j in self.nodes_rel_pos.iteritems():
                    if debug:
                        print 'Trying a path through', i, 'with angle', j[1], 'and distance', j[0]

                    if j[1] < ( (rel_pos_item[1]+90.0) ):
                        if j[1] > ( (rel_pos_item[1]-90.0) ):
                            # This means that the node is going forward to the destination
                            if name_item == i:
                                recep_angle = rel_pos_item[2]
                            else:
                                recep_angle = nodes_geo[name_item].anglewith(nodes_geo[i])

                            if recep_angle < ( (rel_pos_item[2]+self.cone_angle/2.0) ):
                                if recep_angle > ( (rel_pos_item[2]-self.cone_angle/2.0) ):
                                    if inside:
                                        if j[0] > self.cone_radius:
                                            if debug:
                                                print 'case 1'
                                            continue

                                        if j[0] > next_hop_rel_pos[0]:
                                            #if nodes_geo[i].distanceto(nodes_geo[next_hop_name]) < next_dist:
                                            if abs(abs(recep_angle)-abs(rel_pos_item[2])) < abs(abs(next_angle)-abs(rel_pos_item[2])):
                                                next_hop_name = i
                                                next_hop_rel_pos = j
                                                next_dist = distance(nodes_geo[i],nodes_geo[next_hop_name])
                                                next_angle = recep_angle
                                                if debug:
                                                    print 'case 2'                                

                                    else:
                                        if j[0] < self.cone_radius:
                                            inside = True
                                            next_hop_name = i
                                            next_hop_rel_pos = j
                                            next_dist = distance(nodes_geo[i],nodes_geo[next_hop_name])
                                            next_angle = recep_angle
                                            if debug:
                                                print 'case 3'
                                        elif j[0] < next_hop_rel_pos[0]:
                                            next_hop_name = i
                                            next_hop_rel_pos = j
                                            next_dist = distance(nodes_geo[i],nodes_geo[next_hop_name])
                                            next_angle = recep_angle
                                            if debug:
                                                print 'case 4'

            self[name_item] = next_hop_name


        # Apply recursivity
        for i in self:
            while self[i] != self[self[i]]:
                self[i] = self[self[i]]
                if debug:
                    print i, self[i]

        # Check if the power level needed for each next hop allows us to directly reaching destination, even being out of the cone
        for i in self:
            if self.GetLevel(nodes_geo[i])!=None and self.GetLevel(nodes_geo[i])<=self.GetLevel(nodes_geo[self[i]]):
                self[i]=i

        if debug:
            print i, nodes_geo[i], self[i], nodes_geo[self[i]], distance(self.layercake.get_current_position(),nodes_geo[self[i]])


    def BuildRoutingTable2(self):
        # Shortest path with level constraints
        self.logger.debug("Building optimal routing table for node "+self.layercake.hostname)

        G = self.BuildGraph()    
        D,P = self.Dijkstra(G,self.layercake.hostname)

        for i, pos in nodes_geo.iteritems():
            end = i
            if end==self.layercake.hostname:
                continue
            Path = []
            while 1:
                    Path.append(end)
                    if end == self.layercake.hostname: break
                    end = P[end]
            Path.reverse()
            self[i] = Path[1]
            if debug:
                print i, self[i]

    def BuildGraph(self):
        levels = self.layercake.phy.level2distance
        graph = {}
        for fname, fpos in nodes_geo.iteritems():
            graph[fname]={}
            for tname, tpos in nodes_geo.iteritems():
                if fname==tname:
                    continue
                d = distance(fpos,tpos)
                if self.GetLevelFT(d)!=None:
                    # We can reach it using some power level
                    graph[fname][tname] = self.layercake.phy.distance2power[self.GetLevelFT(d)]
           
        return graph


    def GetLevelFT(self, d):
        levels = self.layercake.phy.level2distance
        for level, distance in levels.iteritems():
            if d<=distance:
                return level    


    def Dijkstra(self,G,start,end=None):
        """
        Find shortest paths from the start vertex to all
        vertices nearer than or equal to the end.

        The input graph G is assumed to have the following
        representation: A vertex can be any object that can
        be used as an index into a dictionary.  G is a
        dictionary, indexed by vertices.  For any vertex v,
        G[v] is itself a dictionary, indexed by the neighbors
        of v.  For any edge v->w, G[v][w] is the length of
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

        Of course, G and G[v] need not be Python dict objects;
        they can be any other object that obeys dict protocol,
        for instance a wrapper in which vertices are URLs
        and a call to G[v] loads the web page and finds its links.
        
        The output is a pair (D,P) where D[v] is the distance
        from start to v and P[v] is the predecessor of v along
        the shortest path from s to v.
        
        Dijkstra's algorithm is only guaranteed to work correctly
        when all edge lengths are positive. This code does not
        verify this property for all edges (only the edges seen
        before the end vertex is reached), but will correctly
        compute shortest paths even for some graphs with negative
        edges, and will raise an exception if it discovers that
        a negative edge has caused it to make a mistake.
        """
        D = {}	# dictionary of final distances
        P = {}	# dictionary of predecessors
        Q = priorityDictionary()   # est.dist. of non-final vert.
        Q[start] = 0
        
        for v in Q:
                D[v] = Q[v]
                if v == end: break
                
                for w in G[v]:
                        vwLength = D[v] + G[v][w]
                        if w in D:
                                if vwLength < D[w]:
                                        raise ValueError, \
                                        "Dijkstra: found better path to already-final vertex"
                        elif w not in Q or vwLength < Q[w]:
                                Q[w] = vwLength
                                P[w] = v
        return (D,P)
                            
    def shortestPath(self,start,end):
        """
        Find a single shortest path from the given start vertex
        to the given end vertex.
        The input has the same conventions as Dijkstra().
        The output is a list of the vertices in order along
        the shortest path.
        """
        G = self.BuildGraph()
            
        D,P = self.Dijkstra(G,start,end)
        Path = []
        while 1:
                Path.append(end)
                if end == start: break
                end = P[end]
        Path.reverse()
        return Path


    def NodesPos2Str(self):
        print self.nodes_pos




class FBR(SimpleRoutingTable):
    ''' In this case, DACAP4FBR should be selected as MAC protocol.
        Variation 0: Transmission cone
        Variation 1: Reception cone (transmission cone with big apperture)
    '''
    def __init__(self, layercake, config):
        SimpleRoutingTable.__init__(self, layercake, config)
        nodes_geo[layercake.hostname] = layercake.get_current_position() # Does not make sense with mobile nodes: just for the sinks
        
        self.nodes_pos = {layercake.hostname:layercake.get_current_position()} # Contains already discovered positions
        self.cone_angle = config["coneAngle"] # The cone aperture
        self.incoming_packet = None
        self.packets = set([])
        self.rx_cone = config["rx_cone"]


    def OnPacketReception(self, packet):
        # If this is the final destination of the packet,
        # pass it to the application layer
        # otherwise, send it on.
        if not self.IsDuplicated(packet):
            self.packets.add(packet["ID"])
            self.logger.debug("Processing packet with ID: "+packet["ID"])
            if packet["through"] == packet["dest"]:
                if packet["dest"] == self.layercake.hostname:
                    self.layercake.recv(packet)
                    return True
            else:
                self.SendPacket(packet)
                return False
        

    def SendPacket(self, packet):
        self.incoming_packet = packet
        self.incoming_packet["route"].append((self.layercake.hostname, self.layercake.get_real_current_position()))
        self.incoming_packet["source_position"] = self.layercake.get_current_position() #Current hop position

        if self.incoming_packet["dest"]=="AnySink":
            self.incoming_packet["dest"]=self.CloserSink()
            self.nodes_pos[self.incoming_packet["dest"]] = nodes_geo[self.incoming_packet["dest"]] # This information is known (the sinks)
            self.incoming_packet["dest_position"] = self.nodes_pos[self.incoming_packet["dest"]]
        else:
            # Only Data packets come through this level, and they are already directed to the sinks. I do know the position of the sinks.
            self.nodes_pos[self.incoming_packet["dest"]] = nodes_geo[self.incoming_packet["dest"]]
            self.incoming_packet["dest_position"] = self.nodes_pos[self.incoming_packet["dest"]]            

        if self.IsNeighbor(self.incoming_packet["dest_position"]):
            self.incoming_packet["through"] = self.incoming_packet["dest"]
            self.incoming_packet["through_position"] = self.incoming_packet["dest_position"]
            self.incoming_packet["level"] = 0
        else:
            try:
                self.incoming_packet["through"] = self[self.incoming_packet["dest"]] # If this works, then this is because I know the through
                self.incoming_packet["through_position"] = self.nodes_pos[self.incoming_packet["through"]]
                self.incoming_packet["level"] = self.GetLevel(self.incoming_packet["through_position"])
            except KeyError:
                self.logger.debug("WARNING: route to destination "+self.incoming_packet["dest"]+" not specified for "+self.layercake.hostname+". Starting Discovery Process")
                self.incoming_packet["through"] = "ANY0"
                self.incoming_packet["through_position"] = 0
                self.incoming_packet["level"] = 0
        
        self.layercake.mac.InitiateTransmission(self.incoming_packet)
        

    def IsReachable(self, current_level, dest_pos):
        if self.GetLevel(dest_pos)==None:
            return False
        else:
            return self.GetLevel(dest_pos)<=current_level
            

    def IsNeighbor(self, dest_pos):
        return self.IsReachable(0,dest_pos)


    def NeedExplicitACK(self, current_level, destination):
        if self.layercake.hostname == destination:
            return True

        if self.has_key(destination):
            if self.GetLevel(self.nodes_pos[self[destination]])==None:
                return True
            elif self.GetLevel(self.nodes_pos[self[destination]]) < current_level:
                return True
            else:
                return False
        
        return True

        
    def GetLevel(self, destination):
        levels = self.layercake.phy.level2distance

        for level, r in levels.iteritems():
            if distance(self.layercake.get_current_position(),destination)<=r:
                return level


    def CloserSink(self):
        self.logger.debug("Looking for the closest Sink.")
        self.nodes_rel_pos = {}        
        for name_item, pos_item in nodes_geo.iteritems():
            self.nodes_rel_pos[name_item] = distance(self.layercake.get_current_position(),pos_item)

        b = dict(map(lambda item: (item[1],item[0]),self.nodes_rel_pos.items()))
        min_key = b[min(b.keys())]        
        return min_key
    

    def ImAValidCandidate(self, packet):
        if packet["dest"]==self.layercake.hostname:
            return True
        elif self.rx_cone==0:
            """
            I will be a valid candidate if I am within the transmission cone.
            """
            source_pos = packet["source_position"]
            dest_pos = packet["dest_position"]

            if distance(self.layercake.get_current_position(),dest_pos) < distance(source_pos,dest_pos):
                a = distance(self.layercake.get_current_position(),dest_pos)
                b = distance(source_pos,dest_pos)
                c = distance(source_pos,self.layercake.get_current_position())

                if (b**2+c**2-a**2)/(2*b*c)>0.99 or (b**2+c**2-a**2)/(2*b*c)<-0.99 :
                    A = 0.0
                else:
                    A = math.degrees(math.acos((b**2+c**2-a**2)/(2*b*c)))

                if A<=self.cone_angle/2.0:
                    self.logger.debug("I'm a valid candidate.")
                    return True
                else:
                    self.logger.debug("I'm not a valid candidate.")
                    return False

        elif self.rx_cone==1:
            """
            I will be a valid candidate if I am within the reception cone.
            """
            source_pos = packet["source_position"]
            dest_pos = packet["dest_position"]

            if distance(self.layercake.get_current_position(),dest_pos) < distance(source_pos,dest_pos):
                return True
            else:
                return False


        # This information comes from the Multicast RTS and I may be interested in using it, or not.
##        self.nodes_pos[packet["source"]] = packet["source_position"]
##        self.nodes_pos[packet["dest"]] = packet["dest_position"]
##        self[packet["source"]] = packet["source"] # This is a neighbor


    def AddNode(self, name, pos):
        self.nodes_pos[name] = pos


    def Update(self, dest, dest_pos, through, through_pos):
        self[dest] = through
        self.nodes_pos[dest] = dest_pos


    def IsDuplicated(self, packet):
        if packet["ID"] in self.packets:
            # This packet was already received, I should directly acknowledge it
            self.logger.debug("Discarding a duplicated packet. ID: "+packet["ID"])
            return True

        return False
        

    def SelectRoute(self, candidates, current_through, attemps, destination):
        dist = {}
        ener = {}
        score = {}
        
        if len(candidates) == 0:
            # There have been no answers
            self.logger.debug("Unable to reach "+current_through)

            if self.layercake.phy.CollisionDetected():
                self.logger.debug("There has been a collision, let's give it another chance!")
                return "2CHANCE", self.nodes_pos[current_through]                

            if current_through[0:3]!="ANY":
                #This is not a multicast RTS but a directed RTS which has been not answered
                if attemps < 2:
                    self.logger.debug("Let's give it another chance.")
                    return "2CHANCE", self.nodes_pos[current_through]
                else:                
                    self.logger.debug("Starting multicast selection")
                    return "ANY0",0

            if int(current_through[3])!=(len(self.layercake.phy.level2distance)-1):
                # This is a multicast selection, but not with the maximum transmission power level
                self.logger.debug("Increasing transmission power.")
                level = int(current_through[3])+1

                if self.IsReachable(level,nodes_geo[destination]):
                    self.logger.debug("It is reachable.")
                    return "NEIGH"+str(level),0
                else:
                    self.logger.debug("It is not reachable.")
                    return "ANY"+str(level),0
            else:
                self.logger.debug("Unable to reach any node within any transmission power. ABORTING transmission.")
                return "ABORT",0

        else:
            # There have been answers: for a given transmission power, I should always look for the one that is closer to the destination
            if candidates.has_key(destination):
                return destination, candidates[destination][2]

            # Now without energy criteria, I multiply by zero
            for name, de in candidates.iteritems():
                dist[name] = distance(de[2],nodes_geo[destination])
                ener[name] = de[1]*0.0

            # Average score: min energy and min distance to the final destination
            ee = dict(map(lambda item: (item[1],item[0]),ener.items()))
            max_ee = ee[max(ee.keys())]

            dd = dict(map(lambda item: (item[1],item[0]),dist.items()))
            max_dd = dd[max(dd.keys())]            
            
            for name, de in candidates.iteritems():
                if ener[max_ee]>0:
                    score[name] = dist[name]/dist[max_dd]+ener[name]/ener[max_ee]
                else:
                    score[name] = dist[name]/dist[max_dd]
                
            sc = dict(map(lambda item: (item[1],item[0]),score.items()))
            min_score = sc[min(sc.keys())]

            return min_score, self.nodes_pos[min_score]

