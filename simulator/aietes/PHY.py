from SimPy import Simulation as Sim
import math
from PHYTools import *
from Packet import PHYPacket
import logging

from copy import deepcopy
module_logger=logging.getLogger('AIETES.PHY')

#####################################################################
# Physical Layer
#####################################################################
class PHY():
    '''A Generic Class for Physical interface layers
    '''
    def __init__(self,layercake,channel_event,config):
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
        self.__dict__.update(config)
        self.layercake = layercake

        #Inferred System Parameters
        self.threshold['recieve'] = DB2Linear(ReceivingThreshold(self.frequency, self.bandwidth, self.threshold['SNR']))
        self.threshold['listen'] = DB2Linear(ListeningThreshold(self.frequency, self.bandwidth, self.threshold['LIS']))
        self.ambient_noise = DB2Linear( Noise(self.frequency) + 10*math.log10(self.bandwidth*1e3) ) #In linear scale
        self.interference = self.ambient_noise
        self.collision = False

        #Generate modem/transd etc
        self.modem = Sim.Resource(name='a_modem')
        self.transducer = Transducer(self)
        self.messages = []

    def send(self,FromAbove):
        '''Function called from upper layers to send packet
        '''
        packet=PHYPacket(FromAbove)
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
    def __init__(self, phy,
                 name="a_transducer"):

        Sim.Resource.__init__(self,name=name, capacity=transducer_capacity)

        self.phy = phy
        self.layercake = self.phy.layercake
        self.host = self.layercake.host

        self.transmitting = False
        self.collisions = []
        self.channel_event = self.layercake.channel_event
        self.interference = self.phy.interference

        #Configure event listener
        self.listener = AcousticEventListener(self)
        Sim.activate(
                    self.listener,
                    self.listener.listen(
                        self.channel_event,
                        self.host.vector.getPos)
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
                "%s is not a PHY Packet" % str(packet)
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
            if not doomed \
                and Linear2DB(minSIR) >= self.SIR_thresh \
                and packet.power >= self.phy.threshold['recieve']:
                    # Properly received: enough power, not enough interference
                    self.collision = False
                    self.layercake.mac.recv(new_packet)

            elif packet.power >= self.phy.receiving['recieve']:
                # Too much interference but enough power to receive it: it suffered a collision
                if self.phy.host.name == new_packet.through or self.phy.host.name == new_packet.dest:
                    self.collision = True
                    self.collisions.append(new_packet)
                    self.physical_layer.PrintMessage("A "+new_packet.type+" packet to "+new_packet.through
                                                     +" was discarded due to interference.")
            else:
                # Not enough power to be properly received: just heard.
                self.phy.logger.debug("This packet was not addressed to me.")

        else:
            # This should never appear, and in fact, it doesn't, but just to detect bugs (we cannot have a negative SIR in lineal scale).
            print new_packet.type, new_packet.source, new_packet.dest, new_packet.through, self.physical_layer.host.name


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
            travel_time = distance/speed_of_sound  # Speed of sound in water = 1482.0 m/s

            yield Sim.hold, self, travel_time

            new_incoming_packet = Packet.incoming(transducer.phy, DB2Linear(receive_power), params["packet"])
            Sim.activate(new_incoming_packet,new_incoming_packet.Receive(params["duration"]))


