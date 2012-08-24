from Tools import *
import uuid
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

class Packet(dict):
    '''Base Packet Class
    '''
    def __init__(self,packet,):
        ''' Take the upper level packet and encapsulate it as the payload within the new packet
        while maintaining access to all members.

                    | X | payload|
                    |--->---\/---|
                | Y | X | payload|
                | ------>---\/---|
            | Z | Y | X | payload|

        '''
        if hasattr(packet,'logger'):
            self._start_log(logger=packet.logger)
        else:
            self._start_log()

        #Import everything from the upper packet
        for k,v in packet.__dict__.items():
            if debug: self.logger.debug("Got KV: %s %s"%(k,v))
            setattr(self,k,v)
        self.payload = packet         #Overwrite the 'upper' packet payload with the packet

    def _start_log(self, logger=None):
        if logger is None:
            self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
        else:
            self.logger = logger.getChild("%s"%(self.__class__.__name__))

    def decap(self):
        return self.payload #Should return the upper level packet class

    def isFor(self,node_name):
        ''' Boolean if packet directed at node by either 
        name or broadcast
        '''
        if self.payload.destination == broadcast_address:
            return True
        else:
            return self.payload.destination == node_name

    def __repr__(self):
        return str("To: %s, From: %s, at %d, len(%s)"%(self.destination,self.source,self.launch_time, self.length))

#####################################################################
# Application Packet
#####################################################################
class AppPacket(Packet):
    '''Base Packet Class (AKA AppPacket)
    '''
    source = None
    destination = None
    type = None
    data = None
    length = 24 # Default Packet Length

    def __init__(self, source, dest, pkt_type=None, data=None, route=[], logger=None):
        self._start_log(logger=logger)
        self.source = source
        self.destination = dest
        self.type = pkt_type
        self.launch_time=Sim.now()
        self.route = route
        if data is not None:
            self.data = data
            self.length = len(data)
        else:
            self.data = AppPacket.data
            self.length = AppPacket.length

        self.id=uuid.uuid4() #Hopefully unique id

#####################################################################
# Network Packet
#####################################################################
class RoutePacket(Packet):
    '''Routing Level Packet
    '''

    def __init__(self,route_layer,packet,level=0.0):
        Packet.__init__(self,packet)
        self.level = 0.0
        self.next_hop = None #Analogue to packet['through'] in AUVNetSim
        self.source_position = None
        self.destination_position = None

        self.route.append({'name':route_layer.host.name,
                           'position':route_layer.host.getPos()
                          })
        self.source_position = route_layer.host.getPos()

    def set_level(self,level):
        self.level = level

    def set_next_hop(self,hop):
        self.next_hop=hop

####################################################################
# Media Access Control Packet
#####################################################################
class MACPacket(Packet):
    '''MAC level packet
    '''
    def last_sender(self):
        return self.route[-1]

    def customBehaviour(self):
        #TODO What does a MAC packet do differently?
        pass

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
    def __init__(self, phy, packet=None, power=None):
        Packet.__init__(self,packet)
        Sim.Process.__init__(self, name=self.__class__.__name__)
        self.phy=phy
        self.doomed=False
        self.max_interference=None
        if power is not None:
            self.power = power

    def send(self, power=None):
        # Default power if needed
        if power is None:
            power = self.phy.transmit_power
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

        bitrate = self.phy.bandwidth_to_bit(bandwidth)
        duration = self.length/bitrate

        self.phy.transducer.channel_event.signal({"pos":self.source_position,
                                                  "power":power,
                                                  "duration":duration,
                                                  "frequency":self.phy.frequency,
                                                  "packet":self
                                                 })

        #Lock the transducer for duration of TX
        self.phy.transducer.onTX()
        yield Sim.hold, self, duration
        self.phy.transducer.postTX()

        #Release modem
        yield Sim.release, self, self.phy.modem

        #Update power stats
        power_w = DB2Linear(AcousticPower(power))
        self.phy.tx_energy += (power_w * duration)

        self.logger.info("PHY Packet Sent")

    def recv(self, duration):
        if self.power >= self.phy.threshold['listen']:
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









