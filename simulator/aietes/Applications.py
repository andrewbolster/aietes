from numpy import *
from numpy.random import poisson
import logging
from Tools import baselogger,debug,randomstr, Sim

from Packet import AppPacket

class Application(Sim.Process):
    """
    Generic Class for top level application layers
    """
    def __init__(self,layercake, config=None):
        self._start_log(layercake)
        Sim.Process.__init__(self)
        self.stats={'packets_sent':0,
                    'packets_recieved':0,
                    'packets_time':0,
                    'packets_hops':0,
                    'packets_dhops':0
                   }
        self.packet_log={}
        self.config=config
        self.layercake=layercake
        if hasattr(config,'packet_rate') and getattr(config,'packet_rate') is not None:
            self.packet_rate=getattr(config,'packet_rate')
            self.logger.info("Taking Packet_Rate from config: %s"%self.packet_rate)
        else:
            self.packet_rate = 1
        self.period = 1/float(self.packet_rate)

    def _start_log(self,layercake):
        self.logger = layercake.logger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance')

    def activate(self):
        Sim.activate(self,self.lifecycle())

    def lifecycle(self,destination=None):

        if destination is None:
            self.logger.info("No Destination defined, defaulting to \"AnySink\"")
            destination = "Any"

        while True:
            (packet,period)=self.packetGen(period=self.period,
                                           data=randomstr(24),
                                           destination=destination)
            self.layercake.net.send(packet)
            self.stats['packets_sent'] += 1
            if debug: self.logger.info("Sending Packet: Waiting %s"%period)
            yield Sim.hold, self, period

    def recv(FromBelow):
        """
        Called by RoutingTable on packet reception
        """
        packet = FromBelow.decap()
        self.logPacket(packet)
        self.packetRecv(packet)

    def logPacket(self,packet):
        """
        Grab packet statistics
        """
        assert isinstance(packet,AppPacket)
        source = packet.source
        self.stats['packets_recieved']+=1
        if source in self.packet_log.keys():
            self.packet_log[source].append(packet)
        else:
            self.packet_log[source]=[packet]
        delay = Sim.now() - packet.launch_time
        #TODO Test if this hop check makes sense
        hops = len(FromBelow.route)

        self.stats['packets_time']+=delay
        self.stats['packets_hops']+=hops
        self.stats['packets_dhops']+=(delay/hops)

        self.logger.info("Packet recieved from %s over %d hops with a delay of %s (d/h=%s)"%(source,hops,str(delay),str(delay/hops)))

    def packetGen(self,period,destination):
        """
        Packet Generator with periodicity
        Called from the lifecycle with defaults None,None
        """
        raise TypeError("Tried to instantiate the base Application class")


class AccessibilityTest(Application):
    def packetGen(self,period,destination,data=None):
        """
        Copy of behaviour from AUVNetSim for default class,
        exhibiting poisson departure behaviour
        """
        packet = AppPacket(
            source=self.layercake.host.name,
            dest=destination,
            pkt_type='DATA',
            data=data,
            logger=self.logger
        )
        period=poisson(float(period))
        return (packet,period)

    def packetRecv(self,packet):
        assert isinstance(packet, AppPacket)


