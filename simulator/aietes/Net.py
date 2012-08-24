from Packet import RoutePacket
from Tools import debug

class RoutingTable():
    '''Routing table generic class
    '''
    def __init__(self,layercake,config=None):
        #Generic Spec
        self.logger = layercake.logger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance:%s'%config)

        self.layercake=layercake
        self.host = layercake.host
        self.config=config
        self.has_routing_table=False
        self.table={}

    def send(self,FromAbove):
        packet=RoutePacket(self,FromAbove)
        if not hasattr(self.table,packet.destination):
            packet.set_next_hop(packet.destination)
        else:
            packet.set_next_hop(self.table[packet.destination])
        if debug: self.logger.info("Net Packet Sent")
        self.layercake.mac.send(packet)

    def recv(self,FromBelow):
        if debug: self.logger.info("Net Packet Recieved")
        self.layercake.app.recv(FromBelow.decap())

    def explicitACK(self,FromBelow):
        '''Assume we always want to call for ACK
        i.e. no implicit ACK
        '''
        return True

