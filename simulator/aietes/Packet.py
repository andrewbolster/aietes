from SimPy import Simulation as Sim
import logging
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
module_logger=logging.getLogger('AIETES.PKT')

class Packet():
    '''Base Packet Class
    '''
    def __init__(self,packet):
        ''' Take the upper level packet and encapsulate it as the payload within the new packet
        while maintaining access to all members.

                    | X | payload|
                    |--->---\/---|
                | Y | X | payload|
                | ------>---\/---|
            | Z | Y | X | payload|

        '''
        self._start_log()
        self.__dict__.update(packet)  #Import everything from the upper packet
        self.payload = packet         #Overwrite the 'upper' packet payload with the packet

    def _start_log(self):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')

    def decap(self):
        return self.payload #Should return the upper level packet class

    def isFor(self,node_name):
        return self.payload.destination == node_name

class AppPacket(Packet):
    '''Base Packet Class (AKA AppPacket)
    '''
    self.source = None
    self.destination = None
    self.type = None
    self.data = None
    self.length = 0

    def __init__(self, source, dest, pkt_type, data=None, route=[]):
        self._start_log()
        self.source = node.name
        self.destination = dest
        self.type = pkt_type
        if data is not None:
            self.data = data
            self.length = len(data)



class RoutePacket(Packet):
    '''Routing Level Packet
    '''
    self.level = 0.0
    self.route = []
    self.next_hop = None #Analogue to packet['through'] in AUVNetSim
    self.source_position = None
    self.destination_position = None

    def __init__(self,route_layer,packet,level=0.0):
        Packet.__init__(self,packet)
        self.route.append({'name':route_layer.host.name,
                           'position':route_layer.host.current_postion()
                          })
        self.source_position = None


    def last_sender(self):
        return self.route[-1]

    def set_level(self,level):
        self.level = level

    def set_nexthop(self,hop):
        self.next_hop=hop


class MACPacket(Packet):
    '''MAC level packet
    '''

#####################################################################
# Physical Packet
#####################################################################
class PHYPacket(Sim.Process, Packet):
    '''Generic Physical Layer Packet Class
    Executed as a process and actually performs the send and recieve functions
    for the physical layer.
    This is because the transducer and modem are RESOURCES that are used, not
    processing units.
    '''
    def __init__(self,phy, name):
        Sim.Process.__init__(self, name)
        self.phy=phy
        self.doomed=False

    #TODO THESE WILL NOT WORK
    @classmethod
    def incoming(cls, phy, power, payload):
        '''Overloaded 'Constructor' for incoming packets, recieved 'from' the transducer
        '''
        Packet.__init__(self,payload)
        self.direction = 'in'
        self.power = power
        self.max_interference = 1
        return cls(phy, "RecieveMessage: "+str(payload))

    @classmethod
    def outgoing(cls, phy, power, payload):
        '''Overloaded 'Constructor' for outgoing packets, recieved from the MAC
        '''

        self.direction = 'out'
        self.power = power
        self.payload = payload

    def send(self):
        #Lock Modem
        yield Sim.request, self, self.phy.modem

        #Generate Bandwidth info
        if self.phy.variable_bandwidth:
            distance = self.phy.var_power['levelToDistance'][self.packet.p_level] #TODO Packet description
            bandwidth = distance2Bandwidth(self.power,
                                           self.phy.frequency,
                                           distance,
                                           self.phy.threshold['SNR']
                                          )
        else:
            bandwidth = self.phy.bandwidth

        bitrate = self.phy.bandwidth_to_bit(bandwidth) #TODO Function
        duration = packet.length/bitrate

        self.phy.transducer.channel_event.signal()
        #TODO

        #Lock the transducer for duration of TX
        self.phy.transducer.onTX()
        yield Sim.hold, self, duration
        self.phy.transducer.postTX()

        #Release modem
        yield Sim.release, self, self.phy.modem

        #Update power stats
        power_w = DB2Linear(AcousticPower(self.power))
        self.phy.tx_energy += (power_w * duration)

    def recv(self, duration):
        if self.power >= self.py.threshold['listen']:
            # Otherwise I will not even notice that there are packets in the network
            yield Sim.request, self, self.phy.transducer
            yield Sim.hold, self, duration
            yield Sim.release, self, self.phy.transducer
        
            # Even if a packet is not received properly, we have consumed power
            self.phy.rx_energy += DB2Linear(self.phy.receive_power) * duration #TODO shouldn't this be listen power?

    def updateInterference(self, interference):
        '''A simple ratchet of interference based on the transducer _request func
        '''
        self.max_interference = max(self.max_interference,interference)

    def getMinSIR(self):
        return self.power/(self.max_interference-self.power+1) # TODO BODMAS?

    def Doom(self):
        self.doomed = True









