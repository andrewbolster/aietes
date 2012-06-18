from SimPy import Simulation as Sim
from FSM import FSM
import logging
import pydot
module_logger=logging.getLogger('AIETES.MAC')

class Packet():
    """Generic Class for Packets
    """
    def __init__(self,name="a packet", length=0, signal=None):
        self.name=name
        self.length=length
        self.signal=signal


class MAC():
    """Generic Class for MAC Algorithms
    The only real difference between MAC's are their packet types and State Machines
    """
    #TODO break everything apart so that init does very little and a 
    def __init__(self,config=None):
        self.logger = logging.getLogger("%s.%s"%(module_logger.name,self.__class__.__name__))
        self.logger.info('creating instance')

        self.known_packets=self.packetBuilder()
        self.outgoing_queue=[]
        self.incoming_packet=None
        self.InitialiseStateEngine()

    def activate(self):
        self.logger.info('activating instance')
        self.timer=self.InternalTime(self.sm)                             #Instance of Internal Timer
        self.timer_event = Sim.SimEvent("timer_event")                    #SimPy Event for signalling
        Sim.activate(self.timer, self.timer.Lifecycle(self.timer_event))  #Tie the event to lifecycle function

    def packetBuilder(self,config): 
        """Generate 'known packets' 
        """
        raise TypeError("Tried to instantiate the base MAC class")

    class InternalTimer(Sim.Process):
        """The internal Timer of the MACs is a SimPy Process as well as the nodes themselves.
        In a practical sence, this mirrors the idea that the network interfaces on the Vectors
        operate independently from the 'Command and Control' functionality
        """
        def __init__(self,state_machine):
            Sim.Process.__init__(self,name="%s_Timer"%self.__class__.__name__)
            self.sm=state_machine

        def lifeCycle(self,Request):
            while True:
                yield Sim.waitevent, self, Request
                yield Sim.hold, self, Request.signalparam[1]  #Wait for a given time
                if (self.interrupted()):
                    self.interruptReset()
                else :
                    self.sm.process(Request.signalparam[0])   #Do something


    def InitialiseStateEngine(self):
        """Generate a FSM with an initial READY_WAIT state
        """
        self.sm = FSM("READY_WAIT", []) 
        # Default transition to error state to fallback
        self.sm.set_default_transition(self.onError, "READY_WAIT")

    def send(self, FromAbove):
        """Function Called from upper layers to send a packet
        """
        self.outgoing_queue.append(FromAbove)
        self.sm.process("send_data")

    def transmit(self):
        """Real Transmission of packet to physical layer
        """
        self.layercake.phy.send(self.outgoing_queue[0])

    def onTX_success(self):
        """When an ACK has been recieved, we can assume it all went well
        """
        self.logger.info("Successful TX to "+self.outgoing_queue[0].through) #TODO Packet definition
        self.postTX()

    def onTX_fail(self):
        """When an ACK has timedout, we can assume it is impossible to contact the next hop
        """
        self.logger.info("Timed out TX to "+self.outgoing_queue[0].through)
        self.postTX()

    def postTX(self):
        """Succeeded or given up, either way, tidy up
        """
        self.outgoing_queue.pop(0)
        self.transmission_attempts = 0

        if (len(self.outgoing_queue)>0):
            self.queueNext()

    def recv(self, FromBelow):
        """Function Called from lower layers to recieve a packet
        """
        self.incoming_packet = FromBelow
        if FromBelow.isFor(self.node.name): #TODO Packet Definition Update
            self.sm.process(self.type_to_signal[FromBelow.type])
        else:
            self.overheard()

    def overheard(self):
        pass

    def onRX(self):
        """Recieved a packet
        """
        origin = self.incoming_packet["route"][-1][0] #TODO WTF
        self.logger.info("RX-d packet from %s"%origin)

        #TODO handoff to routing layer

    def onError(self):
        """ Called via state machine when unexpected input symbol in a determined state
        """
        self.logger.err("Unexpected transition by %s from %s because of symbol %s from %s"%
                        (self.node.name, self.sm.current_state, self.sm.input_symbol, self.incoming_packet)
                       )
    def onTimeout(self):
        """When it all goes wrong
        """
        pass

    def queueData(self):
        """Log queueing
        """#TODO what the hell is this?
        self.logger.info("Queueing Data")

    def draw(self):
        self.sm.state_transitions


class ALOHA(MAC):
    """A very simple algorithm
    """
    def packetBuilder(self):
        packets=[]
        packets.append(Packet(
                        name="ACK",
                        length=24,
                        signal="got_ACK"
                        ))
        packets.append(Packet(
                        name="DATA",
                        signal="got_DATA"
                        ))
        return packets

    def InitialiseStateEngine(self):
        """Set up the state machine for ALOHA
        """
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
