from SimPy import Simulation as Sim
import math
from defaults import PHYconfig as defaults
from PHYTools import *
import logging

from copy import deepcopy
module_logger=logging.getLogger('AIETES.PHY')

#####################################################################
# Physical Packet
#####################################################################
class PHYPacket(Sim.Process):
    '''Generic Physical Layer Packet Class
    '''
    def __init__(self,phy, name):
        Sim.Process.__init__(self, name)
        self.phy=phy
        self.doomed=False

    @classmethod
    def incoming(cls, phy, power, payload):
        '''Overloaded 'Constructor' for incoming packets, recieved 'from' the transducer
        '''
        self.direction = 'in'
        self.power = power
        self.payload = payload
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

#####################################################################
# Physical Layer
#####################################################################
class PHY():
    '''A Generic Class for Physical interface layers
    '''
    def __init__(self,name="A_PHY",channel_event):
        '''Initialise with defaults from PHYconfig
        :Frequency Specifications
            frequency (kHz)
            bandwidth (kHz)
            bandwidth_to_bit_ratio (bps/Hz)
            variable_bandwidth (bool)
        :Power Specifications
            transmit_power (dB re 1 uPa @1m)
            max_transmit_power (dB re 1 uPa @1m)
            recieve_power (dBw)
            listen_power (dBw)
            var_power (None|,
                { 'levelToDistance':
                    {'(n)':'(km)',(*)},
                }
            )
        :Initial Detection Specifications
            threshold (
                { 'SIR': (dB),
                  'SNR': (dB),# sufficient to recieve
                  'LIS': (dB) # sufficient to detect
                }
            }
        '''
        #Generic Spec
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')
        self.name=name
        self.__dict__.update(defaults)

        #Inferred System Parameters
        self.threshold['recieve'] = DB2Linear(ReceivingThreshold(self.frequency, self.bandwidth, self.threshold['SNR']))
        self.threshold['listen'] = DB2Linear(ListeningThreshold(self.frequency, self.bandwidth, self.threshold['LIS']))
        self.ambient_noise = DB2Linear( Noise(self.freq) + 10*math.log10(self.bandwidth*1e3) ) #In linear scale
        self.intereference = self.ambient_noise
        self.collision = False

        #Generate modem/transd etc
        self.modem = Sim.Resource(name='a_modem')
        self.transducer = Transducer(self)

        self.messages = []

    def send(self,mac_packet):
        '''Function called from upper layers to send packet
        '''
        packet=PHYPacket(mac_packet.decap())
        if self.IsIdle()==False:
            self.PrintMessage("I should not do this... the channel is not idle!") # The MAC protocol is the one that should check this before transmitting

        self.collision = False

        if self.variable_power:
            distance = self.level2distance[packet.level]
            power = distance2Intensity(self.bandwidth, self.freq, distance, self.SNR_threshold)
        else:
            power = self.transmit_power

        if power > self.max_output_power_used:
            self.max_output_power_used = power

        if power > self.max_output_power:
            power = self.max_output_power

        new_transmission = OutgoingPacket(self)
        Sim.activate(new_transmission, new_transmission.transmit(packet, power))

#####################################################################
# Transducer
#####################################################################
class Transducer(Sim.Resource):
    '''Packets request the resource and can interfere
    '''
    def __init__(self, name="a_transducer",phy, ambient_noise, SIR_threshold,
                 channel_event,
                 pos_query_func,
                 success_callback_func):
        Sim.Resource.__init__(self,name=name, capacity=defaults.transducer_capacity)

        self.phy = phy
        self.SIR_threshold = SIR_threshold
        self.transmitting = False
        self.collisions = []
        self.interference = ambient_noise #Noise==Inter as far as this is concerned
        self.callback = success_callback_func
        self.channel_event = channel_event

        #Configure event listener
        self.listener = AcousticEventListener(self)
        Sim.activate(self.listener,
                     self.listener.listen(self.channel_event, pos_query_func)
                    )

    def updateInterference(self,packet):
        ''' Update interferences of the active queue
        '''
        if self.transmitting:
            packet.Doom()

        self.interference += packet.power

        [x.updateInterference(self.interference) for x in self.activeQ()]

    def _request(self,arg):
        '''Overiding SimPy's to update interference information upon queuing of a new incoming packet from the channel
        '''
        Sim.Resource._request(self,arg)
        #Arg[1] is a reference to the newly queued incoming packet
        self.updateInterference(arg[1])

    # Override SimPy Resource's "_release" function to update SIR for all incoming messages.
    def _release(self, arg):
        # "arg[1] is a reference to the Packet instance that just completed
        packet = arg[1]
        assert isinstance(packet, PHYPacket), \
                "%s is not a PHY Packet", % str(packet)
        doomed = packet.doomed
        minSIR = packet.GetMinSIR()
        mac_packet = deepcopy(packet.payload)

        # Reduce the overall interference by this message's power
        self.interference -= packet.power
        # Prevent rounding errors 
        #TODO shouldn't this be to <= ambient?
        if self.interference<=0:
            self.interference = self.phy.ambient_noise

        # Delete this from the transducer queue by calling the Parent form of "_release"
        Sim.Resource._release(self, arg)

        # If it isn't doomed due to transmission & it is not interfered
        if minSIR>0:
            if not doomed
            and Linear2DB(minSIR) >= self.SIR_thresh
            and packet.power >= self.phy.threshold['recieve']:
                # Properly received: enough power, not enough interference
                self.collision = False
                self.on_success(new_packet)

            elif packet.power >= self.phy.receiving['recieve']:
                # Too much interference but enough power to receive it: it suffered a collision
                if self.phy.node.name == new_packet.through or self.phy.node.name == new_packet.dest:
                     self.collision = True
                     self.collisions.append(new_packet)
                     self.physical_layer.PrintMessage("A "+new_packet.type+" packet to "+new_packet.through
                                                                 +" was discarded due to interference.")
            else:
                # Not enough power to be properly received: just heard.
                self.phy.logger.debug("This packet was not addressed to me.")

        else:
            # This should never appear, and in fact, it doesn't, but just to detect bugs (we cannot have a negative SIR in lineal scale).
            print new_packet.type, new_packet.source, new_packet.dest., new_packet.through., self.physical_layer.node.name


    def onTX(self):
        self.transmitting = True
        # Doom all currently incoming packets to failure.
        [i.Doom() for i in self.activeQ]


    def postTX(self):
        self.transmitting = False





#####################################################################
# Acoustic Event Listener
#####################################################################

class AcousticEventListener(Sim.Process):
    """No physical analog.
    Waits for another node to send something and then activates
    an Arrival Scheduler instance.
    """
    def __init__(self, transducer):
        Sim.Process.__init__(self)
        self.transducer = transducer

    def listen(self, channel_event, position_query):
        while(True):
            yield Sim.waitevent, self, channel_event
            params = channel_event.signalparam
            sched = ArrivalScheduler(name="ArrivalScheduler"+self.name[-1])
            Sim.activate(sched, sched.schedule_arrival(self.transducer, params, position_query()))


#####################################################################
# Arrival Scheduler
#####################################################################

class ArrivalScheduler(Sim.Process):
    """simulates the transit time of a message
    """

    def schedule_arrival(self, transducer, params, pos):
        distance = pos.distanceto(params["pos"])

        if distance > 0.01:  # I should not receive my own transmissions
            receive_power = params["power"] - Attenuation(params["frequency"], distance)
            travel_time = distance/defaults.speed_of_sound  # Speed of sound in water = 1482.0 m/s

            yield Sim.hold, self, travel_time

            new_incoming_packet = Packet.incoming(transducer.phy, DB2Linear(receive_power), params["packet"])
            Sim.activate(new_incoming_packet,new_incoming_packet.Receive(params["duration"]))


