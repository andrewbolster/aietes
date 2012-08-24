import math
from Tools import *
from Packet import PHYPacket
import logging

from copy import deepcopy

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
            receive_power (dBw)
            listen_power (dBw)
            var_power (None|,
                { 'levelToDistance':
                    {'(n)':'(km)',(*)},
                }
            )
        :Initial Detection Specifications
            threshold (
                { 'SIR': (dB),
                  'SNR': (dB),# sufficient to receive
                  'LIS': (dB) # sufficient to detect
                }
            }
        '''
        #Generic Spec
        self.logger = layercake.logger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance:%s'%config)
        #Inherit values from config
        for k,v in config.items():
            self.logger.info("Updating: [%s]=%s"%(k,v))
            setattr(self,k,v)
        self.layercake = layercake

        #Inferred System Parameters
        self.threshold['receive'] = DB2Linear(ReceivingThreshold(self.frequency, self.bandwidth, self.threshold['SNR']))
        self.threshold['listen'] = DB2Linear(ListeningThreshold(self.frequency, self.bandwidth, self.threshold['LIS']))
        self.ambient_noise = DB2Linear( Noise(self.frequency) + 10*math.log10(self.bandwidth*1e3) ) #In linear scale
        self.interference = self.ambient_noise
        self.collision = False

        #Generate modem/transd etc
        self.modem = Sim.Resource(name=self.__class__.__name__)
        self.transducer = Transducer(self)
        self.messages = []

        #Statistics
        self.max_output_power_used = 0
        self.tx_energy = 0
        self.rx_energy = 0

    def isIdle(self):
        """Before TX, check if the transducer activeQ (inherited from Sim.Resource) is empty i.e
        Are we recieving?
        """
        if len(self.transducer.activeQ)>0:
            self.logger.info("The Channel is not idle: %d packets currently in flight"%str(len(self.transducer.activeQ)))
            return False
        return True

    def send(self,FromAbove):
        '''Function called from upper layers to send packet
        '''
        packet = PHYPacket(self,FromAbove)
        if not self.isIdle():
            self.PrintMessage("I should not do this... the channel is not idle!") # The MAC protocol is the one that should check this before transmitting

        self.collision = False

        if hasattr(self,"variable_power") and self.variable_power:
            tx_range = self.level2distance[packet.level]
            power = distance2Intensity(self.bandwidth, self.frequency, tx_range, self.SNR_threshold)
        else:
            self.logger.info("Using Static Power Model")
            power = self.transmit_power

        if power > self.max_output_power_used:
            self.max_output_power_used = power

        Sim.activate(packet, packet.send(power))

    def bandwidth_to_bit(self, bandwidth):
        return bandwidth * 1e3 * self.bandwidth_to_bit_ratio

#####################################################################
# Transducer
#####################################################################
class Transducer(Sim.Resource):
    '''Packets request the resource and can interfere
    '''
    def __init__(self, phy,
                 name="Transducer"):

        Sim.Resource.__init__(self,name=name, capacity=transducer_capacity)
        self.logger = phy.logger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance')

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
                        self.host.getPos)
                    )

    def updateInterference(self,packet):
        ''' Update interferences of the active queue
        '''
        if self.transmitting:
            packet.Doom()

        self.interference += packet.power

        [x.updateInterference(self.interference) for x in self.activeQ]

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
        minSIR = packet.getMinSIR()

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
                and Linear2DB(minSIR) >= self.phy.threshold['SIR'] \
                and packet.power >= self.phy.threshold['receive']:
                    # Properly received: enough power, not enough interference
                    self.collision = False
                    self.layercake.mac.recv(packet.payload)

            elif packet.power >= self.phy.threshold['receive']:
                # Too much interference but enough power to receive it: it suffered a collision
                if self.host.name == packet.next_hop or self.host.name == packet.destination:
                    self.collision = True
                    self.collisions.append(packet)
                    self.physical_layer.PrintMessage("A "+packet.type+" packet to "+packet.next_hop
                                                     +" was discarded due to interference.")
            else:
                # Not enough power to be properly received: just heard.
                self.phy.logger.debug("This packet was not addressed to me.")

        else:
            # This should never appear, and in fact, it doesn't, but just to detect bugs (we cannot have a negative SIR in lineal scale).
            print packet.type, packet.source, packet.dest, packet.next_hop, self.physical_layer.host.name


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
            #Wait until something happens on the channel
            yield Sim.waitevent, self, channel_event

            params = channel_event.signalparam
            if debug: self.transducer.logger.debug("Scheduling Arrival: Params: %s"%params)
            sched = ArrivalScheduler(name="ArrivalScheduler"+self.name[-1])
            Sim.activate(sched, sched.schedule_arrival(self.transducer, params, position_query()))


#####################################################################
# Arrival Scheduler
#####################################################################

class ArrivalScheduler(Sim.Process):
    """simulates the transit time of a message
    """

    def schedule_arrival(self, transducer, params, pos):
        distance_to = distance(pos, params['pos'])

        if distance_to > 0.01:  # I should not receive my own transmissions
            receive_power = params["power"] - Attenuation(params["frequency"], distance_to)
            travel_time = distance_to/speed_of_sound  # Speed of sound in water = 1482.0 m/s

            yield Sim.hold, self, travel_time

            new_incoming_packet = PHYPacket(transducer.phy, power=DB2Linear(receive_power), packet = params["packet"])
            baselogger.info("ArrivalScheduler: Back from Yielding: %s"%travel_time)
            baselogger.info("Type: %s"%(type(new_incoming_packet)))
            Sim.activate(new_incoming_packet, new_incoming_packet.recv(duration=params["duration"]))


