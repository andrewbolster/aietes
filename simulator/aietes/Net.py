
class RoutingTable():
    '''Routing table generic class
    '''
    def __init__(self,layercake):
        self.layercake=layercake
        self.has_routing_table=False
        self.table={}

    def send(self,FromAbove):
        packet=RoutePacket(self,FromAbove)
        if not hasattr(self.table,packet.destination):
            packet.set_nexthop(packet.destination)
        else:
            packet.set_nexthop(self.table[packet.destination])
        self.layercake.mac.send(packet)

    def recv(self,FromBelow):
        self.layercake.app.recv(FromBelow.decap())

    def explicitACK(self,FromBelow):
        '''Assume we always want to call for ACK
        i.e. no implicit ACK
        '''
        return True

