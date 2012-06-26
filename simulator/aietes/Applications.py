import SimPy.Simulation as Sim
from numpy import *
from numpy.random import poisson
import logging

from Packet import AppPacket

module_logger=logging.getLogger('AIETES.MAC')

class Application(Sim.Process):
    """
    Generic Class for top level application layers
    """
    def __init__(self,layercake, config=None):
        self._start_log()
        Sim.Process.__init__(self)
        self.stats={'packets_sent':0,
                    'packets_recieved':0,
                    'packets_time':0,
                    'packets_hops':0,
                    'packets_dhops':0
                   }
        self.packet_log={}
        self.config=config
        self.packet_rate=None
        if hasattr(config,'packet_rate'):
            self.packet_rate=getattr(config,'packet_rate')

    def _start_log(self):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')

    def lifecycle(self,period=self.packet_rate,destination==None):
        while True:
            (packet,period)=self.packetGen(period,destination)
            self.layercake.net.send(packet)
            self.stats['packets_sent']+=1
            yield Sim.hold, self, period

    def packetGen(self,period,destination="AnySink"):
        """
        Copy of behaviour from AUVNetSim for default class,
        exhibiting poisson departure behaviour
        """
        packet = AppPacket(
            source=self.layercake.host.name,
            dest=destination,
            pkt_type='DATA'
        )
        period=numpy.poisson(period)
        return (packet,period)

    def recv(FromBelow):
        """
        Called by RoutingTable on packet reception
        """
        packet = FromBelow.decap()
        self.log_packet(packet)

    def log_packet(self,packet):
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

        self.logger.info("Packet recieved from %s over %d hops with a delay of %s (d/h=%s)"%(source,hops,str(delay),str(delay/hops))



