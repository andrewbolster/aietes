from SimPy import Simulation as Sim
from FSM import FSM
from Packet import Packet
from Tools import dotdict, baselogger
import logging
import pydot

class MAC():
    '''Generic Class for MAC Algorithms
    The only real difference between MAC's are their packet types and State Machines
    '''
    def __init__(self,layercake,config=None):
        self.logger = layercake.logger.getChild("%s"%(self.__class__.__name__))
        self.logger.info('creating instance')
        self.config=config
        self.layercake = layercake

        self.macBuilder()
        self.outgoing_queue=[]
        self.incoming_packet=None
        self.InitialiseStateEngine()

    def activate(self):
        self.logger.info('activating instance')
        self.timer=self.InternalTimer(self.sm)                             #Instance of Internal Timer
        self.timer_event = Sim.SimEvent("timer_event")                    #SimPy Event for signalling
        Sim.activate(self.timer, self.timer.lifecycle(self.timer_event))  #Tie the event to lifecycle function

    def macBuilder(self):
        '''Generate run-time MAC config
        '''
        raise TypeError("Tried to instantiate the base MAC class")

    class InternalTimer(Sim.Process):
        '''The internal Timer of the MACs is a SimPy Process as well as the nodes themselves.
        In a practical sence, this mirrors the idea that the network interfaces on the Vectors
        operate independently from the 'Command and Control' functionality
        '''
        def __init__(self,state_machine):
            Sim.Process.__init__(self,name="%s_Timer"%self.__class__.__name__)
            self.sm=state_machine

        def lifecycle(self,Request):
            while True:
                yield Sim.waitevent, self, Request
                yield Sim.hold, self, Request.signalparam[1]  #Wait for a given time
                if (self.interrupted()):
                    self.interruptReset()
                else :
                    self.sm.process(Request.signalparam[0])   #Do something


    def InitialiseStateEngine(self):
        '''Generate a FSM with an initial READY_WAIT state
        '''
        self.sm = FSM("READY_WAIT", []) 
        # Default transition to error state to fallback
        self.sm.set_default_transition(self.onError, "READY_WAIT")

    def send(self, FromAbove):
        '''Function Called from upper layers to send a packet
        Encapsulates the Route layer packet in to a MAC Packet
        '''
        self.outgoing_queue.append(MACPacket(FromAbove))
        self.sm.process("send_data")

    def transmit(self):
        '''Real Transmission of packet to physical layer
        '''
        self.layercake.phy.send(self.outgoing_queue[0])

    def onTX_success(self):
        '''When an ACK has been recieved, we can assume it all went well
        '''
        self.logger.info("Successful TX to "+self.outgoing_queue[0].next_hop) 
        self.postTX()

    def onTX_fail(self):
        '''When an ACK has timedout, we can assume it is impossible to contact the next hop
        '''
        self.logger.info("Timed out TX to "+self.outgoing_queue[0].next_hop)
        self.postTX()

    def postTX(self):
        '''Succeeded or given up, either way, tidy up
        '''
        self.outgoing_queue.pop(0)
        self.transmission_attempts = 0

        if (len(self.outgoing_queue)>0):
            self.queueNext()

    def recv(self, FromBelow):
        '''Function Called from lower layers to recieve a packet
        Decapsulates the packet from the physical
        '''
        self.incoming_packet = FromBelow.decap()
        if FromBelow.isFor(self.node.name):
            self.sm.process(self.type_to_signal[FromBelow.type])
        else:
            self.overheard()

    def overheard(self):
        pass

    def onRX(self):
        '''Recieved a packet
        Should ack, but can drop ack if routing layer says its ok
        Sends packet up to higher level
        '''
        origin = self.incoming_packet.last_sender()
        self.logger.info("RX-d packet from %s"%origin)

        if self.layercake.net.explicitACK(self.incoming_packet):
            #TODO Make ACK
            ack=generateACK(self.incoming_packet)
            self.transmit(ack)
            # Send up to next level in stack
            self.layercake.net.recv(self.incoming_packet)



    def onError(self):
        ''' Called via state machine when unexpected input symbol in a determined state
        '''
        self.logger.err("Unexpected transition by %s from %s because of symbol %s from %s"%
                        (self.node.name, self.sm.current_state, self.sm.input_symbol, self.incoming_packet)
                       )
    def onTimeout(self):
        '''When it all goes wrong
        '''
        pass

    def queueData(self):
        '''Log queueing
        '''#TODO what the hell is this?
        self.logger.info("Queueing Data")

    def draw(self):
        graph = pydot.Dot(graph_type = 'digraph')

        #create base state nodes (set of destination states)
        basestyle=''
        states={}
        for state in set(zip(*tuple(self.sm.state_transitions.values()))[1]):
            states[state]=pydot.Node(state,*basestyle)
            graph.add_node(states[state])

        #create function 'nodes'
        functions={}
        for function in  set(zip(*tuple(self.sm.state_transitions.values()))[0]):
            functions[function.__name__]=pydot.Node(function.__name__,shape="parallelogram")
            graph.add_node(functions[function.__name__])

        for (signal,state) in self.sm.state_transitions:
            (function,next_state) = self.sm.state_transitions[(signal,state)]
            graph.add_edge(pydot.Edge(states[state],functions[function.__name__],label=signal))
            #differently formatted return edge
            graph.add_edge(pydot.Edge(functions[function.__name__],states[next_state],style='dotted',shape='onormal'))

        graph.write_png("%s.png"%self.__class__.__name__)

class ALOHA(MAC):
    '''A very simple algorithm
    '''
    def macBuilder(self):
        self.packets=[]
        self.packets.append(dotdict({
            'name':"ACK",
            'length':self.config.ack_packet_length,
            'signal':"got_ACK"
        }))
        self.packets.append(dotdict({
            'name':"DATA",
            'signal':"got_DATA"
        }))

        #Adapted/derived variables
        self.timeout = 0    # co-adapted with TX pwr
        self.level = 0      # derived from routing layer
        self.T = 0



    def InitialiseStateEngine(self):
        '''Set up the state machine for ALOHA
        '''
        MAC.InitialiseStateEngine(self)
        #Transitions from READY_WAIT
        self.sm.add_transition("got_DATA","READY_WAIT", self.onRX, "READY_WAIT")
        self.sm.add_transition("send_DATA","READY_WAIT", self.transmit, "READY_WAIT")

        #Transitions from WAIT_ACK
        self.sm.add_transition("got_DATA", "WAIT_ACK", self.onRX, "WAIT_ACK")
        self.sm.add_transition("send_DATA", "WAIT_ACK", self.queueData, "WAIT_ACK")
        self.sm.add_transition("got_ACK", "WAIT_ACK", self.onTX_success, "READY_WAIT")
        self.sm.add_transition("timeout", "WAIT_ACK", self.onTimeout, "WAIT_2_RESEND")

        #Transitions from WAIT_2_RESEND
        self.sm.add_transition("resend", "WAIT_2_RESEND", self.transmit, "WAIT_2_RESEND")
        self.sm.add_transition("got_DATA", "WAIT_2_RESEND", self.onRX, "WAIT_2_RESEND")
        self.sm.add_transition("send_DATA", "WAIT_2_RESEND", self.queueData, "WAIT_2_RESEND")
        self.sm.add_transition("fail", "WAIT_2_RESEND", self.onTX_fail, "READY_WAIT")

    def overheard(self):
        '''Steal some information from overheard packets
        i.e. implicit ACK: Source hears forwarding transmission of its own packet
        '''
        if self.incoming_packet.type == "DATA" and self.sm.current_state == "WAIT_ACK":
            last_hop=self.incoming_packet.route[-1][0]
            if self.outgoing_queue[0].next_hop == last_hop and self.outgoing_queue[0].id == self.incoming_packet.id:
                self.logger.info("Recieved an implicit ACK from routing node %s"%last_hop)
                self.sm.process("got_ACK")


        #TODO Expand/Duplicate with overhearing position information
