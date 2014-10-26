## -*- coding: cp1252 -*-
###########################################################################
#
# Copyright (C) 2007 by Justin Eskesen and Josep Miquel Jornet Montana
#    <jge@mit.edu> <jmjornet@mit.edu>                                      
#
# Copyright: See COPYING file that comes with this distribution
#
# This file is part of AUVNetSim, a library for simulating acoustic
# networks of fixed and mobile underwater nodes, written in Python.
#
# AUVNetSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AUVNetSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AUVNetSim.  If not, see <http://www.gnu.org/licenses/>.
#
###########################################################################

import SimPy.Simulation as Sim
from FSM import FSM
import random

from aietes.Tools import broadcast_address, np, debug

debug = True


def SetupMAC(node, config):
    if config["protocol"] == "DACAP":
        return DACAP(node, config)
    elif config["protocol"] == "DACAP4FBR":
        return DACAP4FBR(node, config)
    elif config["protocol"] == "CSMA":
        return CSMA(node, config)
    elif config["protocol"] == "CSMA4FBR":
        return CSMA4FBR(node, config)
    elif config["protocol"] == "ALOHA":
        return ALOHA(node, config)
    elif config["protocol"] == "ALOHA4FBR":
        return ALOHA4FBR(node, config)
    else:
        return ALOHA(node, config)


class ALOHA:
    """ALOHA:  A very simple MAC Algorithm
    """

    def __init__(self, layercake, config):
        self.layercake = layercake
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)

        self.ack_packet_length = config["ack_packet_length"]
        self.packet_signal = {"ACK": "got_ACK", "DATA": "got_DATA"}

        self.InitialiseStateEngine()

        self.channel_access_retries = 0  # Number of times that the channel was sensed and found not idle
        self.transmission_attempts = 0  # Number of retransmissions
        self.max_transmission_attempts = config["attempts"]
        self.max_wait_to_retransmit = config["max2resend"]

        self.timeout = 0  # It is adapted with the transmission power level
        self.level = 0  # It will be defined from the routing layer
        self.T = 0  # It will be adapted once the level is fixed

        self.outgoing_packet_queue = []
        self.incoming_packet = None

        self.stats = {"data_packets_sent": 0, "data_packets_received": {}}

        self.timer = self.InternalTimer(self.fsm)
        self.TimerRequest = Sim.SimEvent("TimerRequest")

    def activate(self):
        Sim.activate(self.timer, self.timer.Lifecycle(self.TimerRequest))


    class InternalTimer(Sim.Process):
        def __init__(self, fsm):
            Sim.Process.__init__(self, name="MAC_Timer")
            random.seed()
            self.fsm = fsm

        def Lifecycle(self, Request):
            while True:
                yield Sim.waitevent, self, Request
                yield Sim.hold, self, Request.signalparam[0]
                if (self.interrupted()):
                    self.interruptReset()
                else:
                    self.fsm.process(Request.signalparam[1])


    def InitialiseStateEngine(self):
        """InitialiseStateEngine:  set up Finite State Machine for ALOHA
        """
        self.fsm = FSM("READY_WAIT", [], self)

        #Set default to Error
        self.fsm.set_default_transition(self.OnError, "READY_WAIT")

        #Transitions from READY_WAIT
        self.fsm.add_transition("got_DATA", "READY_WAIT", self.OnDataReception, "READY_WAIT")
        self.fsm.add_transition("send_DATA", "READY_WAIT", self.Transmit, "READY_WAIT")

        #Transitions from WAIT_ACK
        self.fsm.add_transition("got_DATA", "WAIT_ACK", self.OnDataReception, "WAIT_ACK")
        self.fsm.add_transition("send_DATA", "WAIT_ACK", self.QueueData, "WAIT_ACK")
        self.fsm.add_transition("got_ACK", "WAIT_ACK", self.OnTransmitSuccess, "READY_WAIT")
        self.fsm.add_transition("timeout", "WAIT_ACK", self.OnTimeout, "WAIT_2_RESEND")

        #Transitions from WAIT_2_RESEND
        self.fsm.add_transition("resend", "WAIT_2_RESEND", self.Transmit, "WAIT_2_RESEND")
        self.fsm.add_transition("got_DATA", "WAIT_2_RESEND", self.OnDataReception, "WAIT_2_RESEND")
        self.fsm.add_transition("send_DATA", "WAIT_2_RESEND", self.QueueData, "WAIT_2_RESEND")
        self.fsm.add_transition("fail", "WAIT_2_RESEND", self.OnTransmitFail, "READY_WAIT")


    def InitiateTransmission(self, OutgoingPacket):
        ''' Function called from the upper layers to transmit a packet.
        '''
        self.outgoing_packet_queue.append(OutgoingPacket)
        self.fsm.process("send_DATA")


    def OnNewPacket(self, IncomingPacket):
        ''' Function called from the lower layers when a packet is received.
        '''
        self.incoming_packet = IncomingPacket
        if self.IsForMe():
            if debug: self.logger.debug("received Packet from {src}".format(src=IncomingPacket['source']))
            self.fsm.process(self.packet_signal[IncomingPacket["type"]])
        else:
            if debug: self.logger.debug("Overheard Packet from {src} to {dest}".format(src=IncomingPacket['source'], dest=IncomingPacket['dest']))
            self.OverHearing()


    def IsForMe(self):
        if self.layercake.hostname == self.incoming_packet["through"]:
            return True
        else:
            return broadcast_address == self.incoming_packet["through"]


    def OverHearing(self):
        ''' Valuable information can be obtained from overhearing the channel.
        '''
        if self.incoming_packet["type"] == "DATA" and self.fsm.current_state == "WAIT_ACK":
            packet_origin = self.incoming_packet["route"][-1][0]
            if packet_origin == self.outgoing_packet_queue[0]["through"] and self.incoming_packet["dest"] == self.outgoing_packet_queue[0]["dest"]:
                if self.incoming_packet["ID"] == self.outgoing_packet_queue[0]["ID"]:
                    # This is an implicit ACK
                    self.logger.info("An implicit ACK has arrived.")
                    self.fsm.process("got_ACK")


    def OnDataReception(self):
        ''' A data packet is received. We should acknowledge the previous node or we can try to use implicit
            acknowledges if that does not mean a waste of power.
        '''
        packet_origin = self.incoming_packet["route"][-1][0]  # These ACKs just gone to the previous hop, this is maintenance at MAC layer

        if self.layercake.net.NeedExplicitACK(self.incoming_packet["level"], self.incoming_packet["dest"]) \
                or len(self.outgoing_packet_queue) != 0 \
                or self.layercake.net.IsDuplicated(self.incoming_packet):
            self.SendAck(packet_origin)

        self.layercake.net.OnPacketReception(self.incoming_packet)


    def SendAck(self, packet_origin):
        if debug: self.logger.debug("ACK to " + packet_origin)
        AckPacket = {"type": "ACK",
                     "source": self.layercake.hostname,
                     "source_position": self.layercake.get_current_position(),
                     "dest": packet_origin,
                     "dest_position": self.incoming_packet["source_position"],
                     "through": packet_origin,
                     "through_position": self.incoming_packet["source_position"],
                     "length": self.ack_packet_length,
                     "level": self.incoming_packet["level"],
                     "ID": self.incoming_packet["ID"]}

        self.layercake.phy.TransmitPacket(AckPacket)


    def OnError(self):
        ''' This function is called when the FSM has an unexpected input_symbol in a determined state.
        '''
        self.logger.error("Unexpected transition from {sm.last_state} to {sm.current_state} due to {sm.input_symbol}: {pkt}".format(sm=self.fsm, pkt=self.incoming_packet))


    def OnTransmitSuccess(self):
        ''' When an ACK is received, we can assume that everything has gone fine, so it's all done.
        '''
        if debug: self.logger.info("Successfully Transmitted to " + self.incoming_packet["source"])

        #We got an ACK, stop the timer...
        p = Sim.Process()
        p.interrupt(self.timer)

        self.PostSuccessOrFail()


    def OnTransmitFail(self):
        ''' All the transmission attemps have been completed. It's impossible to reach the node.
        '''
        self.logger.error("Failed to transmit to {through} with id {id}".format(
            through=self.outgoing_packet_queue[0]["through"],
            id=self.outgoing_packet_queue[0]["ID"])
        )
        self.PostSuccessOrFail()


    def PostSuccessOrFail(self):
        ''' Successfully or not, we have finished the current transmission.
        '''
        self.outgoing_packet_queue.pop(0)
        self.transmission_attempts = 0
        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))


    def OnTimeout(self):
        self.transmission_attempts += 1
        if debug: self.logger.debug("Timed Out, No Ack Received")

        if self.transmission_attempts > self.max_transmission_attempts:
            self.fsm.process("fail")
        else:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "resend"))


    def Transmit(self):
        ''' Real Transmission of the Packet.
        '''
        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2delay(self.level)
        self.t_data = self.outgoing_packet_queue[0]["length"] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
        self.timeout = 2 * self.T + 2 * self.t_data

        # Before transmitting, we check if the channel is idle        
        if self.layercake.phy.IsIdle():
            self.transmission_attempts = self.transmission_attempts + 1
            self.channel_access_retries = 0
            if debug: self.logger.debug("Transmit to " + self.outgoing_packet_queue[0]["through"])
            self.layercake.phy.TransmitPacket(self.outgoing_packet_queue[0])
            self.fsm.current_state = "WAIT_ACK"
            self.TimerRequest.signal((self.timeout, "timeout"))
            if debug: self.logger.debug("The timeout is " + str(self.timeout))
        else:
            self.channel_access_retries = self.channel_access_retries + 1
            timeout = random.random() * (2 * self.T)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))


    def QueueData(self):
        self.logger.info("Queuing Data")


    def PrintMessage(self, msg):
        pass
        #print "ALOHA (%s): %s at t=" % (self.layercake.hostname, msg), Sim.now(), self.fsm.input_symbol, self.fsm.current_state


class ALOHA4FBR(ALOHA):
    ''' CS-ALOHA adapted for the FBR protocol.
    '''

    def __init__(self, node, config):

        ALOHA.__init__(self, node, config)

        # New packet types
        self.packet_signal["RTS"] = "got_RTS"  # Route Request packet
        self.packet_signal["CTS"] = "got_CTS"  # Route Proposal packet

        self.rts_packet_length = config["rts_packet_length"]
        self.cts_packet_length = config["cts_packet_length"]

        self.valid_candidates = {}
        self.original_through = None

        # New Transitions from READY_WAIT - RTS packets are used to retrieve a route
        self.fsm.add_transition("send_DATA", "READY_WAIT", self.RouteCheck, "READY_WAIT")
        self.fsm.add_transition("send_RTS", "READY_WAIT", self.SendRTS, "READY_WAIT")
        self.fsm.add_transition("transmit", "READY_WAIT", self.Transmit, "READY_WAIT")

        self.fsm.add_transition("got_RTS", "READY_WAIT", self.ProcessRTS, "READY_WAIT")
        self.fsm.add_transition("send_CTS", "READY_WAIT", self.SendCTS, "READY_WAIT")
        self.fsm.add_transition("ignore_RTS", "READY_WAIT", self.IgnoreRTS, "READY_WAIT")

        self.fsm.add_transition("got_ACK", "READY_WAIT", self.IgnoreACK, "READY_WAIT")

        # New State in which we wait for the routes
        self.fsm.add_transition("got_DATA", "WAIT_CTS", self.OnDataReception, "WAIT_CTS")
        self.fsm.add_transition("send_DATA", "WAIT_CTS", self.QueueData, "WAIT_CTS")

        self.fsm.add_transition("got_RTS", "WAIT_CTS", self.IgnoreRTS, "WAIT_CTS")

        self.fsm.add_transition("got_CTS", "WAIT_CTS", self.AppendCTS, "WAIT_CTS")
        self.fsm.add_transition("timeout", "WAIT_CTS", self.SelectCTS, "WAIT_CTS")
        self.fsm.add_transition("transmit", "WAIT_CTS", self.Transmit, "WAIT_CTS")
        self.fsm.add_transition("retransmit", "WAIT_CTS", self.SendRTS, "WAIT_CTS")
        self.fsm.add_transition("send_RTS", "WAIT_CTS", self.SendRTS, "WAIT_CTS")

        self.fsm.add_transition("abort", "WAIT_CTS", self.OnTransmitFail, "READY_WAIT")
        self.fsm.add_transition("got_ACK", "WAIT_CTS", self.OnTransmitSuccess, "WAIT_CTS")

        # New Transitions from WAIT_ACK
        self.fsm.add_transition("got_RTS", "WAIT_ACK", self.IgnoreRTS, "WAIT_ACK")
        self.fsm.add_transition("got_CTS", "WAIT_ACK", self.IgnoreCTS, "WAIT_ACK")

        # New Transitions from WAIT_2_RESEND
        self.fsm.add_transition("resend", "WAIT_2_RESEND", self.RouteCheck, "WAIT_2_RESEND")
        self.fsm.add_transition("send_RTS", "WAIT_2_RESEND", self.SendRTS, "WAIT_2_RESEND")
        self.fsm.add_transition("transmit", "WAIT_2_RESEND", self.Transmit, "WAIT_2_RESEND")

        self.fsm.add_transition("got_RTS", "WAIT_2_RESEND", self.ProcessRTS, "WAIT_2_RESEND")
        self.fsm.add_transition("send_CTS", "WAIT_2_RESEND", self.SendCTS, "WAIT_2_RESEND")
        self.fsm.add_transition("ignore_RTS", "WAIT_2_RESEND", self.IgnoreRTS, "WAIT_2_RESEND")
        self.fsm.add_transition("got_ACK", "WAIT_2_RESEND", self.OnTransmitSuccess, "WAIT_2_RESEND")


    def IgnoreRTS(self):
        self.logger.debug("Ignoring RTS received from " + self.incoming_packet["source"])
        self.incoming_packet = None


    def IgnoreCTS(self):
        self.logger.debug("Ignoring CTS coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def IgnoreACK(self):
        self.logger.debug("Ignoring ACK coming from " + self.incoming_packet["through"])
        print self.layercake.hostname, "Have we had a collision?"
        self.incoming_packet = None


    def SendRTS(self):
        ''' The RTS sent is the normal one, but we should initialize the list of replies.
        '''
        self.valid_candidates = {}

        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2distance[self.level] / 1482.0

        if self.layercake.phy.IsIdle():
            self.transmission_attempts = self.transmission_attempts + 1
            self.channel_access_retries = 0

            if self.outgoing_packet_queue[0]["through"] == broadcast_address:
                self.multicast = True
            else:
                self.multicast = False

            RTSPacket = {"type": "RTS", "ID": self.outgoing_packet_queue[0]["ID"],
                         "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                         "dest": self.outgoing_packet_queue[0]["dest"], "dest_position": self.outgoing_packet_queue[0]["dest_position"],
                         "through": self.outgoing_packet_queue[0]["through"], "through_position": self.outgoing_packet_queue[0]["through_position"],
                         "length": self.rts_packet_length, "level": self.outgoing_packet_queue[0]["level"], "time_stamp": Sim.now()}

            if debug: self.logger.debug("Transmitting RTS to " + self.outgoing_packet_queue[0]["dest"] + " through " + self.outgoing_packet_queue[0]["through"] + " with power level " + str(self.outgoing_packet_queue[0]["level"]))

            self.layercake.phy.TransmitPacket(RTSPacket)

            self.level = self.outgoing_packet_queue[0]["level"]
            self.T = self.layercake.phy.level2distance[self.level] / 1482.0
            timeout = 2 * self.T

            self.TimerRequest.signal((timeout, "timeout"))
            self.fsm.current_state = "WAIT_CTS"

        else:  #I'm currently not limiting the number of channel access retries
            self.logger.info("The channel was not idle.")
            self.channel_access_retries = self.channel_access_retries + 1
            timeout = random.random() * (2 * self.T + self.t_data)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))


    def ProcessRTS(self):
        ''' Someone is looking for help, may I help? Now the active nodes are the ones that transmit, maybe we should do it the opposite way
        '''
        if self.fsm.current_state != "WAIT_ACK":
            if self.layercake.net.ImAValidCandidate(self.incoming_packet):
                if self.layercake.net.IsDuplicated(self.incoming_packet):
                    self.SendACK(self.incoming_packet["source"])
                    self.fsm.current_state = "READY_WAIT"
                else:
                    self.multicast = True
                    self.fsm.process("send_CTS")
            else:
                self.logger.info("I can't attend the RTS received from " + self.incoming_packet["source"] + ".")
                self.fsm.process("ignore_RTS")


    def RouteCheck(self):
        if self.outgoing_packet_queue[0]["through"] == broadcast_address:
            self.fsm.process("send_RTS")
        else:
            self.fsm.process("transmit")


    def SendCTS(self):
        ''' Clear To Send: I'm proposing myself as a good candidate for the next transmission or I just let transmit if I have been
        already selected.
        '''
        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.layercake.phy.IsIdle():
            CTSPacket = {"type": "CTS",
                         "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                         "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                         "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                         "length": self.cts_packet_length, "rx_energy": self.layercake.phy.rx_energy,
                         "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

            if debug: self.logger.info("Transmitting CTS to " + self.incoming_packet["source"])
            self.layercake.phy.TransmitPacket(CTSPacket)

        self.incoming_packet = None


    def AppendCTS(self):
        ''' More than one CTS is received when looking for the next best hop. We should consider all of them.
        The routing layer decides.
        '''
        self.valid_candidates[self.incoming_packet["through"]] = (Sim.now() - self.incoming_packet["time_stamp"]) / 2.0, self.incoming_packet["rx_energy"], self.incoming_packet["through_position"]
        self.logger.info("Appending CTS to " + self.incoming_packet["source"] + " coming from " + self.incoming_packet["through"])
        self.layercake.net.AddNode(self.incoming_packet["through"], self.incoming_packet["through_position"])

        self.incoming_packet = None

        if self.multicast == False:
            # Update the timer: 1.-Stop, 2.-Restart
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("timeout")


    def SelectCTS(self):
        ''' Once we have wait enough, that is, 2 times the distance at which the best next hop should be, we should select it.
        '''

        current_through = self.outgoing_packet_queue[0]["through"]
        self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0]["through_position"] = self.layercake.net.SelectRoute(self.valid_candidates, self.outgoing_packet_queue[0]["through"], self.transmission_attempts, self.outgoing_packet_queue[0]["dest"])

        if self.outgoing_packet_queue[0]["through"] == "ABORT":
            # We have consumed all the attemps
            self.fsm.process("abort")

        elif self.outgoing_packet_queue[0]["through"] == broadcast_address:
            # We should retransmit increasing the power
            self.outgoing_packet_queue[0]["level"] = int(self.outgoing_packet_queue[0]["through"][3])
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "2CHANCE":
            # We should give a second chance to current node
            self.outgoing_packet_queue[0]["through"] = current_through
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "NEIGH":
            # With current transmission power level, the destination has become a neighbor
            self.outgoing_packet_queue[0]["through"] = self.outgoing_packet_queue[0]["dest"]
            self.outgoing_packet_queue[0]["through_position"] = self.outgoing_packet_queue[0]["dest_position"]
            self.fsm.process("retransmit")

        else:
            self.fsm.process("transmit")


    def Transmit(self):
        ''' Real Transmission of the Packet.
        '''
        self.layercake.net.Update(self.outgoing_packet_queue[0]["dest"], self.outgoing_packet_queue[0]["dest_position"], self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0]["through_position"])
        ALOHA.Transmit(self)


    def CanIHelp(self, packet):
        ''' A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        '''
        # Is this a packet already "directed" to me? I'm not checking if only dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTS and ACK are the only types of packet that are directly addressed
        if packet["type"] == "CTS" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "ACK" and packet["dest"] == self.layercake.hostname:
            return True

        # Is this a multicast packet?
        if packet["through"][0:3] == "ANY":
            return True

        return False


    def OnNewPacket(self, IncomingPacket):
        ''' Function called from the lower layers when a packet is received.
        '''
        self.incoming_packet = IncomingPacket

        if self.CanIHelp(self.incoming_packet):
            if self.incoming_packet["through"][0:3] == "ANY":
                self.fsm.process("got_RTS")
            else:
                self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            ALOHA.OverHearing(self)


    def PrintMessage(self, msg):
        pass
        #print "ALOHA4FBR (%s): %s at t=" % (self.layercake.hostname, msg), Sim.now(), self.fsm.input_symbol, self.fsm.current_state


class DACAP:
    """DACAP : Distance Aware Collision Avoidance Protocol coupled with power control
    """

    def __init__(self, layercake, config):
        self.layercake = layercake
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)

        self.InitialiseStateEngine()
        self.timer = self.InternalTimer(self.fsm)
        self.TimerRequest = Sim.SimEvent("TimerRequest")

        self.outgoing_packet_queue = []
        self.incoming_packet = None
        self.last_packet = None

        self.packet_signal = {"ACK": "got_ACK", "RTS": "got_RTS", "CTS": "got_CTS", "DATA": "got_DATA", "WAR": "got_WAR"}
        self.ack_packet_length = config["ack_packet_length"]
        self.rts_packet_length = config["rts_packet_length"]
        self.cts_packet_length = config["cts_packet_length"]
        self.WAR_packet_length = config["WAR_packet_length"]

        self.max_transmission_attempts = config["attempts"]

        self.transmission_attempts = 0
        self.max_wait_to_retransmit = config["max2resend"]
        self.channel_access_retries = 0
        self.next_timeout = 0
        self.pending_packet_ID = None

        # DACAP specific parameters
        self.T = 0  # It will be adapted according to the transmission range

        self.t_min = config["tmin/T"]
        self.Tw_min = config["twmin/T"]

        self.t_data = self.node.config["packet_length"] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
        self.t_control = self.rts_packet_length / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)

        self.deltaTData = config["deltatdata"]
        self.deltaD = config["deltad/T"]

    def activate(self):
        Sim.activate(self.timer, self.timer.Lifecycle(self.TimerRequest))


    class InternalTimer(Sim.Process):
        def __init__(self, fsm):
            Sim.Process.__init__(self, name="MAC_Timer")
            random.seed()
            self.fsm = fsm

        def Lifecycle(self, Request):
            while True:
                yield Sim.waitevent, self, Request
                yield Sim.hold, self, Request.signalparam[0]
                if (self.interrupted()):
                    self.interruptReset()
                else:
                    if not Request.occurred:
                        self.fsm.process(Request.signalparam[1])


    def InitialiseStateEngine(self):
        """InitialiseStateEngine:  set up Finite State Machine for RTSCTS
        """
        self.fsm = FSM("READY_WAIT", [], self)

        #Set default to Error
        self.fsm.set_default_transition(self.OnError, "READY_WAIT")

        #Transitions from READY_WAIT
        # Normal transitions
        self.fsm.add_transition("send_DATA", "READY_WAIT", self.SendRTS, "READY_WAIT")
        self.fsm.add_transition("got_RTS", "READY_WAIT", self.CheckRTS, "READY_WAIT")
        self.fsm.add_transition("got_X", "READY_WAIT", self.XOverheard, "BACKOFF")
        # Strange but possible transitions
        self.fsm.add_transition("got_DATA", "READY_WAIT", self.CheckPendingData, "READY_WAIT")
        self.fsm.add_transition("got_ACK", "READY_WAIT", self.CheckPendingACK, "READY_WAIT")
        self.fsm.add_transition("got_WAR", "READY_WAIT", self.IgnoreWAR, "READY_WAIT")
        self.fsm.add_transition("got_CTS", "READY_WAIT", self.IgnoreCTS, "READY_WAIT")

        #Transitions from WAIT_CTS
        # Normal transitions
        self.fsm.add_transition("send_DATA", "WAIT_CTS", self.QueueData, "WAIT_CTS")
        self.fsm.add_transition("got_CTS", "WAIT_CTS", self.ProcessCTS, "WAIT_TIME")
        self.fsm.add_transition("got_RTS", "WAIT_CTS", self.IgnoreRTS, "WAIT_CTS")
        self.fsm.add_transition("got_X", "WAIT_CTS", self.XOverheard, "WAIT_CTS")
        self.fsm.add_transition("timeout", "WAIT_CTS", self.OnCTSTimeout, "READY_WAIT")
        self.fsm.add_transition("backoff", "WAIT_CTS", self.XOverheard, "BACKOFF")
        self.fsm.add_transition("got_ACK", "WAIT_CTS", self.OnTransmitSuccess, "READY_WAIT")  # It was a duplicated packet.
        # Strange but possible transitions:
        self.fsm.add_transition("got_WAR", "WAIT_CTS", self.XOverheard, "BACKOFF")  # The CTS had collided, that's why I'm still here.

        #Transitions from WAIT_TIME
        self.fsm.add_transition("send_DATA", "WAIT_TIME", self.QueueData, "WAIT_TIME")
        self.fsm.add_transition("got_RTS", "WAIT_TIME", self.IgnoreRTS, "WAIT_TIME")
        self.fsm.add_transition("got_WAR", "WAIT_TIME", self.ProcessWAR, "WAIT_TIME")
        self.fsm.add_transition("got_ACK", "WAIT_TIME", self.IgnoreACK, "WAIT_TIME")

        self.fsm.add_transition("got_X", "WAIT_TIME", self.XOverheard, "WAIT_TIME")
        self.fsm.add_transition("timeout", "WAIT_TIME", self.Transmit, "WAIT_ACK")
        self.fsm.add_transition("backoff", "WAIT_TIME", self.XOverheard, "BACKOFF")

        #Transitions from WAIT_DATA: I can receive RTS from other nodes, the DATA packet that I expect or overhear other communications
        self.fsm.add_transition("send_DATA", "WAIT_DATA", self.QueueData, "WAIT_DATA")
        self.fsm.add_transition("got_RTS", "WAIT_DATA", self.XOverheard, "WAIT_DATA")
        self.fsm.add_transition("got_DATA", "WAIT_DATA", self.OnDataReception, "READY_WAIT")

        self.fsm.add_transition("got_X", "WAIT_DATA", self.XOverheard, "WAIT_DATA")
        self.fsm.add_transition("timeout", "WAIT_DATA", self.OnDATATimeout, "READY_WAIT")
        self.fsm.add_transition("backoff", "WAIT_DATA", self.XOverheard, "BACKOFF")

        #Transitions from WAIT_ACK: I can receive RTS from other nodes, the ACK packet that I expect or overhear other communications
        self.fsm.add_transition("send_DATA", "WAIT_ACK", self.QueueData, "WAIT_ACK")
        self.fsm.add_transition("got_RTS", "WAIT_ACK", self.IgnoreRTS, "WAIT_ACK")
        self.fsm.add_transition("got_ACK", "WAIT_ACK", self.OnTransmitSuccess, "READY_WAIT")
        self.fsm.add_transition("got_WAR", "WAIT_ACK", self.ProcessWAR, "WAIT_ACK")  # This should not happen

        self.fsm.add_transition("got_X", "WAIT_ACK", self.XOverheard, "BACKOFF")
        self.fsm.add_transition("timeout", "WAIT_ACK", self.OnACKTimeout, "READY_WAIT")

        #Transitions from BACKOFF
        self.fsm.add_transition("send_DATA", "BACKOFF", self.QueueData, "BACKOFF")
        self.fsm.add_transition("got_RTS", "BACKOFF", self.ProcessRTS, "BACKOFF")
        self.fsm.add_transition("got_CTS", "BACKOFF", self.IgnoreCTS, "BACKOFF")
        self.fsm.add_transition("got_DATA", "BACKOFF", self.OnDataReception, "BACKOFF")
        self.fsm.add_transition("got_ACK", "BACKOFF", self.OnTransmitSuccess, "READY_WAIT")
        self.fsm.add_transition("got_WAR", "BACKOFF", self.ProcessWAR, "BACKOFF")

        self.fsm.add_transition("got_X", "BACKOFF", self.XOverheard, "BACKOFF")
        self.fsm.add_transition("timeout", "BACKOFF", self.OnTimeout, "READY_WAIT")
        self.fsm.add_transition("accept", "BACKOFF", self.SendCTS, "WAIT_DATA")
        self.fsm.add_transition("backoff", "BACKOFF", self.XOverheard, "BACKOFF")


    def XOverheard(self):
        ''' By overhearing the channel, we can obtain valuable information.
        '''
        if self.fsm.current_state == "READY_WAIT":
            if self.incoming_packet["type"] != "RTS" and self.incoming_packet["type"] != "CTS":
                self.logger.info("I'm in READY_WAIT and have received " + self.incoming_packet["type"])

        elif self.fsm.current_state == "WAIT_CTS":
            if self.incoming_packet["type"] == "CTS":
                k = self.T * self.t_min
            elif self.incoming_packet["type"] == "RTS":
                k = self.T
            else:
                # Let's think: Can I overhear something else? I think I shouldn't
                self.logger.info("I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
                return False

            if (Sim.now() - self.time) < k:
                self.fsm.process("backoff")
                return True
            else:
                return False

        elif self.fsm.current_state == "WAIT_TIME":
            if self.incoming_packet["type"] == "CTS":
                k = self.T * self.t_min
            elif self.incoming_packet["type"] == "RTS":
                k = self.T
            else:
                # Let's think: Can I overhear something else? I think I shouldn't
                self.logger.info("I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
                return True

            if (Sim.now() - self.time) < k:
                self.fsm.process("backoff")
                return True
            else:
                return False

        elif self.fsm.current_state == "WAIT_DATA":
            if self.incoming_packet["type"] == "RTS" and self.incoming_packet["ID"] == self.pending_packet_ID:
                # It is the same RTS or the same with a higher power value - It doesn't make sense but may happen
                return True
            else:
                if self.incoming_packet["type"] == "RTS":
                    k = 2 * self.T - self.T * self.t_min
                elif self.incoming_packet["type"] == "CTS":
                    k = 2 * self.T - self.T * self.Tw_min
                else:
                    # Let's think: Can I overhear something else? I think I shouldn't
                    self.logger.info("I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
                    return True

                if (Sim.now() - self.time) < k:
                    self.SendWarning()
                    self.fsm.process("backoff")
                    return True
                else:
                    return False

        T = self.layercake.phy.level2delay(self.level)
        self.last_packet = self.incoming_packet

        # If I am here, that means that I already was in the backoff state. I have just to schedule the backoff timer
        if self.incoming_packet["type"] == "WAR":
            backoff = self.BackOff(self.incoming_packet["type"], T)  # This is new!
        elif self.incoming_packet["type"] == "CTS" or self.incoming_packet["type"] == "SIL":
            backoff = self.BackOff(self.incoming_packet["type"], T)
        elif self.incoming_packet["type"] == "RTS" or self.incoming_packet["type"] == "DATA":
            backoff = self.BackOff(self.incoming_packet["type"], T)
        elif self.incoming_packet["type"] == "ACK":
            backoff = 0.0  # I'm done

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        self.logger.info("Sleep for " + str(backoff) + " due to " + self.incoming_packet["type"]
                         + " coming from " + self.incoming_packet["source"]
                         + " to " + self.incoming_packet["dest"]
                         + " through " + self.incoming_packet["through"])

        self.TimerRequest.signal((backoff, "timeout"))
        self.next_timeout = Sim.now() + backoff

        self.incoming_packet = None


    def IgnoreRTS(self):
        self.logger.info("I can't attend the RTS received from " + self.incoming_packet["source"])
        self.incoming_packet = None


    def IgnoreCTS(self):
        self.logger.info("Ignoring CTS coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def IgnoreACK(self):
        self.logger.info("Ignoring ACK coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def IgnoreWAR(self):
        self.logger.info("Ignoring WAR coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def CheckPendingData(self):
        if self.pending_packet_ID != None:
            if self.incoming_packet["ID"] == self.pending_packet_ID:
                self.logger.warn("Despite everything, I properly received the DATA packet from: " + self.incoming_packet["source"])
                self.OnDataReception()
        else:
            self.OnError()


    def CheckPendingACK(self):
        if self.pending_packet_ID != None:
            if self.incoming_packet["ID"] == self.pending_packet_ID:
                self.logger.warn("Despite everything, I we properly transmitted to: " + self.incoming_packet["source"])
                self.OnTransmitSuccess()
        else:
            self.OnError()


    def SendRTS(self):
        ''' Request To Send.
        '''
        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2distance[self.level] / 1482.0

        if self.layercake.phy.IsIdle():
            self.transmission_attempts = self.transmission_attempts + 1
            self.channel_access_retries = 0

            if self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
                self.multicast = True
            else:
                self.multicast = False

            RTSPacket = {"type": "RTS", "ID": self.outgoing_packet_queue[0]["ID"],
                         "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                         "dest": self.outgoing_packet_queue[0]["dest"], "dest_position": self.outgoing_packet_queue[0]["dest_position"],
                         "through": self.outgoing_packet_queue[0]["through"], "through_position": self.outgoing_packet_queue[0]["through_position"],
                         "length": self.rts_packet_length, "level": self.outgoing_packet_queue[0]["level"], "time_stamp": Sim.now()}

            if debug: self.logger.info("Transmitting RTS to " + self.outgoing_packet_queue[0]["dest"] + " through " + self.outgoing_packet_queue[0]["through"] + " with power level " + str(self.outgoing_packet_queue[0]["level"]))

            self.level = self.outgoing_packet_queue[0]["level"]
            self.T = self.layercake.phy.level2distance[self.level] / 1482.0

            self.layercake.phy.TransmitPacket(RTSPacket)

            timeout = self.TimeOut("CTS", self.T)
            self.TimerRequest.signal((timeout, "timeout"))
            self.time = Sim.now()
            self.pending_packet_ID = self.outgoing_packet_queue[0]["ID"]
            self.fsm.current_state = "WAIT_CTS"
        else:
            self.channel_access_retries = self.channel_access_retries + 1
            timeout = random.random() * (2 * self.T + self.t_data)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))


    def ProcessRTS(self):
        ''' Maybe I can do it now.
        '''
        if self.last_packet["type"] == "CTS" and self.last_packet["through"] == self.incoming_packet["source"]:
            # I was sleeping because one of my neighbors was receiving a packet and now it wants to forward it.
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")
        elif self.last_packet["type"] == "DATA" and self.last_packet["through"] == self.incoming_packet["source"] and self.last_packet["dest"] == self.incoming_packet["dest"]:
            # I was sleeping because one of my neighbors was receiving a packet which I also overheard.
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")


    def TimeWait(self, T, U):
        ''' Returns the time to wait before transmitting
        '''
        t1 = (self.t_min * T - min(self.deltaD * T, self.t_data, 2 * T - self.t_min * T)) / 2.0
        t2 = (self.t_min * T - self.deltaTData) / 2.0
        t3 = (self.t_min * T + self.Tw_min * T - 2 * self.deltaD * T) / 4.0

        if U >= t1 and U <= t2:
            timewait = 2 * (U + self.deltaD * T) - self.t_min * T
            self.logger.info("First case, U=" + str(U) + ", I'll wait for " + str(timewait))
        elif U > max(t2, min(t1, t3)):
            timewait = 2 * (U + self.deltaD * T) - self.Tw_min * T
            self.logger.info("Second case, U=" + str(U) + ", I'll wait for " + str(timewait))
        else:
            timewait = self.t_min * T - 2 * U
            self.logger.info("Third case, U=" + str(U) + ", I'll wait for " + str(timewait))

        if timewait <= max(2 * self.deltaD * T, self.Tw_min * T):
            self.logger.info("Fourth case, U=" + str(U) + ", I'll wait for " + str(max(2 * self.deltaD * T, self.Tw_min * T)))
            return max(2 * self.deltaD * T, self.Tw_min * T)
        else:
            return timewait


    def BackOff(self, packet_type, T):
        ''' Returns the backoff for a specific state.
        '''
        if packet_type == "RTS" or packet_type == "WAR":
            backoff = 2 * T + 2 * T - self.Tw_min * T
        elif packet_type == "CTS" or packet_type == "SIL":
            backoff = 2 * T + 2 * T - self.Tw_min * T + self.t_data
        elif packet_type == "DATA":
            backoff = 2 * T

        backoff = np.random.random() * (backoff)

        self.logger.warning("Backoff for {backoff} for {type} packet based on T{T},Tw{TW},Td{TD}".format(
            backoff=backoff,
            type=packet_type,
            T=T,
            TW=self.Tw_min,
            TD=self.t_data))
        return backoff


    def TimeOut(self, packet_type, T):
        ''' Returns the timeout for a specific state.
        '''
        if packet_type == "CTS":
            return 2 * T + 2 * self.t_control
        elif packet_type == "DATA":
            return 2 * T + 2 * T - self.Tw_min * T + 2 * self.t_data + 2 * self.t_control
        elif packet_type == "ACK":
            return 2 * T + self.t_data + self.t_control


    def ProcessCTS(self):
        ''' The CTS has been received.
        '''
        self.logger.info("CTS from " + self.incoming_packet["source"] + " coming from " + self.incoming_packet["through"] + " properly received.")

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        U = (Sim.now() - self.incoming_packet["time_stamp"]) / 2.0
        timewait = self.TimeWait(self.T, U)

        self.logger.info("Waiting for " + str(timewait) + " due to " + self.incoming_packet["type"]
                         + " coming from " + self.incoming_packet["source"]
                         + " through " + self.incoming_packet["through"])

        self.TimerRequest.signal((timewait, "timeout"))

        self.incoming_packet = None


    def SendWarning(self):
        ''' If the next best hop has been already selected but a CTS has been received, we should indicate to the sender that
            it is not necessary anymore. Otherwise, implicit WAR can be used.
        '''
        WARPacket = {"type": "WAR",
                     "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                     "dest": self.last_cts["dest"], "dest_position": self.last_cts["dest_position"],
                     "through": self.last_cts["dest"], "through_position": self.last_cts["dest_position"],
                     "length": self.WAR_packet_length, "level": self.last_cts["level"]}

        self.logger.info("Transmitting Warning Packet to " + self.last_cts["dest"])

        self.layercake.phy.TransmitPacket(WARPacket)


    def ProcessWAR(self):
        ''' I should defer the packets for the same receiver.
        '''
        if self.fsm.current_state == "WAIT_ACK":
            self.logger.info("Well, in any case, I wait for the ACK as I have just transmitted.")
            self.incoming_packet = None
            return True
        elif self.fsm.current_state != "READY_WAIT" and self.fsm.current_state is not "WAIT_ACK":
            self.logger.info("Ok, then maybe later on we will talk about this " + self.incoming_packet["source"])
            self.fsm.process("backoff")


    def CheckRTS(self):
        ''' Before proceeding with the data transmission, just check if it's a duplicated packet (maybe the ACK collided).
        '''
        if self.layercake.net.IsDuplicated(self.incoming_packet):
            packet_origin = [self.incoming_packet["source"], self.incoming_packet["source_position"]]
            self.SendACK(packet_origin)
        else:
            self.SendCTS()
            self.fsm.current_state = "WAIT_DATA"


    def SendCTS(self):
        ''' Clear To Send: I'm proposing myself as a good candidate for the next transmission or I just let transmit if I have been
        already selected.
        '''
        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        CTSPacket = {"type": "CTS",
                     "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                     "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                     "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                     "length": self.cts_packet_length, "rx_energy": self.layercake.phy.rx_energy,
                     "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"],
                     "ID": self.incoming_packet["ID"]}

        if debug: self.logger.info("Transmitting CTS to " + self.incoming_packet["source"])
        self.pending_packet_ID = self.incoming_packet["ID"]
        self.last_cts = CTSPacket  # I may need this if I have to send a warning packet
        self.layercake.phy.TransmitPacket(CTSPacket)

        self.level = self.incoming_packet["level"]
        self.T = self.layercake.phy.level2distance[self.level] / 1482.0

        timeout = self.TimeOut("DATA", self.T)
        self.TimerRequest.signal((timeout, "timeout"))
        self.time = Sim.now()

        self.incoming_packet = None


    def SendACK(self, packet_origin):
        ''' Sometimes we can not use implicit ACKs.
        '''
        if debug: self.logger.debug("ACK to {}".format(packet_origin[0]))

        AckPacket = {"type": "ACK",
                     "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                     "dest": packet_origin[0], "dest_position": packet_origin[1],
                     "through": packet_origin[0], "through_position": packet_origin[1],
                     "length": self.ack_packet_length, "level": self.incoming_packet["level"],
                     "ID": self.incoming_packet["ID"]}

        self.layercake.phy.TransmitPacket(AckPacket)


    def InitiateTransmission(self, OutgoingPacket):
        ''' Function called from the upper layers to transmit a packet.
        '''
        self.outgoing_packet_queue.append(OutgoingPacket)
        self.fsm.process("send_DATA")


    def OnNewPacket(self, IncomingPacket):
        ''' Function called from the lower layers when a packet is received.
        '''
        self.incoming_packet = IncomingPacket
        if self.CanIHelp(IncomingPacket):
            self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            self.OverHearing()


    def CanIHelp(self, packet):
        ''' A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        '''
        # Is this a packet already "directed" to me? I'm not checking if only dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTSs are the only types of packet that are directly addressed
        if packet["type"] == "CTS" and packet["dest"] == self.layercake.hostname:
            return True

        return False


    def OverHearing(self):
        ''' Valuable information can be obtained from overhearing the channel.
        '''
        if self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet["dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")
        else:
            self.fsm.process("got_X")


    def OnDataReception(self):
        ''' After the RTS/CTS exchange, a data packet is received. We should acknowledge the previous node or we can try to use implicit
        acknowledges if the that does not mean a waste of power.
        '''
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.fsm.current_state != "BACKOFF":
            self.last_packet = None

        # We should acknowledge the previous node, this is not an END to END ACK
        packet_origin = self.incoming_packet["route"][-1]

        if self.layercake.net.NeedExplicitACK(self.incoming_packet["level"], self.incoming_packet["dest"]) or len(self.outgoing_packet_queue) != 0:
            self.SendACK(packet_origin)

        self.layercake.net.OnPacketReception(self.incoming_packet)

        self.pending_packet_ID = None
        self.incoming_packet = None


    def OnError(self):
        ''' An unexpected transition has been followed. This should not happen.
        '''
        self.logger.info("ERROR!")
        print self.layercake.hostname, self.incoming_packet, self.fsm.input_symbol, self.fsm.current_state


    def OnTransmitSuccess(self):
        ''' When an ACK is received, we can assume that everything has gone fine, so it's all done.
        '''
        if debug: self.logger.info("Successfully Transmitted to " + self.incoming_packet["source"])
        self.pending_packet_ID = None

        # We got an ACK, we should stop the timer.
        p = Sim.Process()
        p.interrupt(self.timer)

        self.PostSuccessOrFail()


    def OnTransmitFail(self):
        ''' All the transmission attemps have been completed. It's impossible to reach the node.
        '''
        self.logger.info("Failed to transmit to " + self.outgoing_packet_queue[0]["dest"])
        self.PostSuccessOrFail()


    def PostSuccessOrFail(self):
        ''' Successfully or not, we have finished the current transmission.
        '''
        self.outgoing_packet_queue.pop(0)["dest"]
        self.transmission_attempts = 0

        # Is there anything else to do?
        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0


    def OnACKTimeout(self):
        ''' The timeout has experied and NO ACK has been received.
        '''
        self.transmission_attempts += 1
        self.logger.info("Timed Out, No Ack Received")

        if self.transmission_attempts < self.max_transmission_attempts:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))


    def OnDATATimeout(self):
        ''' The timeout has experied and NO DATA has been received.
        '''
        self.logger.info("Timed Out!, No Data Received")

        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0


    def OnCTSTimeout(self):
        self.transmission_attempts += 1
        self.logger.info("Timed Out, No CTS Received")
        if self.layercake.phy.CollisionDetected():
            self.logger.info("It seems that there has been a collision.")

        if self.transmission_attempts > self.max_transmission_attempts:
            self.logger.info("Sorry, I cannot do anything else.")
        else:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))


    def OnTimeout(self):
        self.logger.info("Exiting from back off")

        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0


    def Transmit(self):
        self.logger.info("Transmit to " + self.outgoing_packet_queue[0]["dest"] + " through " + self.outgoing_packet_queue[0]["through"])
        self.layercake.phy.TransmitPacket(self.outgoing_packet_queue[0])

        timeout = self.TimeOut("ACK", self.T)
        self.TimerRequest.signal((timeout, "timeout"))


    def QueueData(self):
        self.logger.info("Queuing Data")


class DACAP4FBR(DACAP):
    '''DACAP for FBR: adaptation of DACAP with power control to couple it with FBR
    '''

    def __init__(self, node, config):

        DACAP.__init__(self, node, config)

        # A new packet type
        self.packet_signal["SIL"] = "got_SIL"
        self.SIL_packet_length = config["SIL_packet_length"]

        self.valid_candidates = {}
        self.original_through = None

        # By MC_RTS we refer to MultiCast RTS packets. We should consider to process them.
        self.fsm.add_transition("got_MC_RTS", "READY_WAIT", self.ProcessMCRTS, "AWARE")
        self.fsm.add_transition("got_SIL", "READY_WAIT", self.IgnoreSIL, "READY_WAIT")
        self.fsm.add_transition("defer", "READY_WAIT", self.IgnoreWAR, "READY_WAIT")

        # Definition of a new state, AWARE, and its transitions
        self.fsm.add_transition("send_CTS", "AWARE", self.SendCTS, "WAIT_DATA")
        self.fsm.add_transition("ignore_RTS", "AWARE", self.XOverheard, "BACKOFF")

        # Now I am receiving several CTS, then, I should just append them
        self.fsm.add_transition("got_MC_RTS", "WAIT_CTS", self.ProcessMCRTS, "WAIT_CTS")  # Maybe I should tell him to be silent
        self.fsm.add_transition("got_CTS", "WAIT_CTS", self.AppendCTS, "WAIT_CTS")
        self.fsm.add_transition("got_WAR", "WAIT_CTS", self.ProcessWAR, "WAIT_CTS")
        self.fsm.add_transition("got_SIL", "WAIT_CTS", self.ProcessSIL, "BACKOFF")

        self.fsm.add_transition("timeout", "WAIT_CTS", self.SelectCTS, "WAIT_CTS")
        self.fsm.add_transition("transmit", "WAIT_CTS", self.ProcessCTS, "WAIT_TIME")
        self.fsm.add_transition("retransmit", "WAIT_CTS", self.SendRTS, "WAIT_CTS")
        self.fsm.add_transition("abort", "WAIT_CTS", self.OnTransmitFail, "READY_WAIT")
        self.fsm.add_transition("defer", "WAIT_CTS", self.XOverheard, "BACKOFF")

        # It may be the case that I still receive CTS when being in WAIT_TIME
        self.fsm.add_transition("got_WAR", "WAIT_TIME", self.ProcessWAR, "WAIT_TIME")
        self.fsm.add_transition("got_SIL", "WAIT_TIME", self.ProcessSIL, "BACKOFF")
        self.fsm.add_transition("got_MC_RTS", "WAIT_TIME", self.ProcessMCRTS, "WAIT_TIME")
        self.fsm.add_transition("got_CTS", "WAIT_TIME", self.IgnoreCTS, "WAIT_TIME")
        self.fsm.add_transition("timeout", "WAIT_TIME", self.SelectCTS, "WAIT_TIME")
        self.fsm.add_transition("transmit", "WAIT_TIME", self.Transmit, "WAIT_ACK")
        self.fsm.add_transition("retransmit", "WAIT_TIME", self.SendRTS, "WAIT_CTS")
        self.fsm.add_transition("abort", "WAIT_TIME", self.OnTransmitFail, "READY_WAIT")
        self.fsm.add_transition("defer", "WAIT_TIME", self.XOverheard, "BACKOFF")

        # New transition from WAIT_DATA: I have been not selected as the next hop
        self.fsm.add_transition("got_MC_RTS", "WAIT_DATA", self.ProcessMCRTS, "WAIT_DATA")
        self.fsm.add_transition("ignored", "WAIT_DATA", self.XOverheard, "BACKOFF")
        self.fsm.add_transition("got_CTS", "WAIT_DATA", self.IgnoreCTS, "WAIT_DATA")
        self.fsm.add_transition("got_WAR", "WAIT_DATA", self.IgnoreWAR, "WAIT_DATA")  # From maybe previous transmissions - check it
        self.fsm.add_transition("got_SIL", "WAIT_DATA", self.IgnoreSIL, "WAIT_DATA")

        self.fsm.add_transition("got_WAR", "BACKOFF", self.IgnoreWAR, "BACKOFF")  # From maybe previous transmissions - check it
        self.fsm.add_transition("got_MC_RTS", "BACKOFF", self.ProcessMCRTS, "BACKOFF")
        self.fsm.add_transition("got_SIL", "BACKOFF", self.IgnoreSIL, "BACKOFF")
        self.fsm.add_transition("defer", "BACKOFF", self.XOverheard, "BACKOFF")

        self.fsm.add_transition("got_MC_RTS", "WAIT_ACK", self.ProcessMCRTS, "WAIT_ACK")


    def IgnoreSIL(self):
        self.logger.info("Ignoring SIL coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def ProcessMCRTS(self):
        ''' Someone is looking for help, may I help? Now the active nodes are the ones that transmit, maybe we should do it the opposite way
        '''
        if self.fsm.current_state == "AWARE":
            if self.layercake.net.ImAValidCandidate(self.incoming_packet):
                if self.layercake.net.IsDuplicated(self.incoming_packet):
                    packet_origin = [self.incoming_packet["source"], self.incoming_packet["source_position"]]
                    self.SendACK(packet_origin)
                    self.fsm.current_state = "READY_WAIT"
                else:
                    self.fsm.process("send_CTS")
            else:
                self.logger.info("I can't attend the MultiCast RTS received from " + self.incoming_packet["source"] + " but I will be silent.")
                self.fsm.process("ignore_RTS")
                # The SIlence part is being revised.
                ##        elif self.fsm.current_state == "WAIT_DATA":
                ##            if self.last_cts["dest"] == self.incoming_packet["source"]:
                ##                self.IgnoreRTS()
                ##            else:
                ##                self.SendSilence()
                ##        elif self.fsm.current_state == "WAIT_CTS" or self.fsm.current_state == "WAIT_TIME":
                ##            self.SendSilence()
                ##        elif self.fsm.current_state == "BACKOFF":


    def SendSilence(self):
        ''' Please be quiet!
        '''
        SILPacket = {"type": "SIL",
                     "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                     "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                     "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                     "length": self.SIL_packet_length, "tx_energy": self.layercake.phy.tx_energy,
                     "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

        self.logger.info("Transmitting SIL to " + self.incoming_packet["source"])

        self.layercake.phy.TransmitPacket(SILPacket)

        self.incoming_packet = None


    def ProcessSIL(self):
        ''' The next time that I try to find a route I should do it starting again from ANY0.
        '''
        self.outgoing_packet_queue[0]["through"] = "ANY0"
        self.outgoing_packet_queue[0]["level"] = 0
        self.XOverheard()


    def AppendCTS(self):
        ''' More than one CTS is received when looking for the next best hop. We should consider all of them.
        The routing layer decides.
        '''
        self.valid_candidates[self.incoming_packet["through"]] = (Sim.now() - self.incoming_packet["time_stamp"]) / 2.0, self.incoming_packet["rx_energy"], self.incoming_packet["through_position"]
        self.logger.info("Appending CTS to " + self.incoming_packet["source"] + " coming from " + self.incoming_packet["through"])
        self.layercake.net.AddNode(self.incoming_packet["through"], self.incoming_packet["through_position"])

        self.incoming_packet = None

        if self.multicast == False:
            # Update the timer: 1.-Stop, 2.-Restart
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("timeout")


    def SelectCTS(self):
        ''' Once we have wait enough, that is, 2 times the distance at which the best next hop should be, we should select it.
        '''
        current_through = self.outgoing_packet_queue[0]["through"]
        self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0]["through_position"] = self.layercake.net.SelectRoute(self.valid_candidates, self.outgoing_packet_queue[0]["through"], self.transmission_attempts, self.outgoing_packet_queue[0]["dest"])

        if self.outgoing_packet_queue[0]["through"] == "ABORT":
            # We have consumed all the attemps
            self.fsm.process("abort")

        elif self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
            # We should retransmit increasing the power
            self.outgoing_packet_queue[0]["level"] = int(self.outgoing_packet_queue[0]["through"][3])
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "2CHANCE":
            # We should give a second chance to current node
            self.outgoing_packet_queue[0]["through"] = current_through
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"][0:5] == "NEIGH":
            # With current transmission power level, the destination has become a neighbor
            self.outgoing_packet_queue[0]["level"] = int(self.outgoing_packet_queue[0]["through"][5])
            self.outgoing_packet_queue[0]["through"] = self.outgoing_packet_queue[0]["dest"]
            self.outgoing_packet_queue[0]["through_position"] = self.outgoing_packet_queue[0]["dest_position"]
            self.fsm.process("retransmit")

        else:
            self.fsm.process("transmit")


    def ProcessCTS(self):
        ''' Once a candidate has been selected, we should wait before transmitting according to DACAP.
        '''
        U = self.valid_candidates[self.outgoing_packet_queue[0]["through"]][0]

        if self.multicast:
            timewait = self.TimeWait(self.T, U) - (2 * self.T - U)  # Or zero...
        else:
            timewait = self.TimeWait(self.T, U)

        self.logger.info("Waiting for " + str(timewait) + " before Transmitting through " + self.outgoing_packet_queue[0]["through"])
        self.TimerRequest.signal((timewait, "timeout"))


    def Transmit(self):
        ''' Real Transmission of the Packet.
        '''
        self.layercake.net.Update(self.outgoing_packet_queue[0]["dest"], self.outgoing_packet_queue[0]["dest_position"], self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0]["through_position"])
        DACAP.Transmit(self)


    def OverHearing(self):
        ''' Valuable information can be obtained from overhearing the channel.
        '''
        if self.incoming_packet["type"] == "DATA" and self.fsm.current_state == "WAIT_DATA":
            if self.pending_packet_ID == self.incoming_packet["ID"]:
                # I have not been selected as the next hop. I read it from the data.
                self.fsm.process("ignored")

        elif self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet["dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")

        elif self.incoming_packet["type"] == "CTS" and self.fsm.current_state == "WAIT_DATA" and self.last_cts["dest"] == self.incoming_packet["dest"]:
            # This is another candidate proposing himself as a good candidate
            self.fsm.process("got_CTS")

        elif self.incoming_packet["type"] == "SIL":
            return False

        else:
            self.fsm.process("got_X")


    def CanIHelp(self, packet):
        ''' A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        '''
        # Is this a packet already "directed" to me? I'm not checking if only dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTS, ACK, WAR and SIL are the only types of packet that are directly addressed
        if packet["type"] == "CTS" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "ACK" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "WAR" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "SIL" and packet["dest"] == self.layercake.hostname:
            return True

        # Is this a multicast packet?
        if packet["through"][0:3] == "ANY":
            return True

        return False


    def OnNewPacket(self, IncomingPacket):
        ''' Function called from the lower layers when a packet is received.
        '''
        self.incoming_packet = IncomingPacket

        if self.CanIHelp(self.incoming_packet):
            if self.incoming_packet["through"][0:3] == "ANY":
                self.fsm.process("got_MC_RTS")
            else:
                self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            self.OverHearing()


    def IgnoreWAR(self):
        ''' If I'm already in back off due to a SIL packet, I just ignore the WAR packets that I receive.
        '''
        self.logger.info("Ok, Ok, I already knew that " + self.incoming_packet["source"])
        self.incoming_packet = None


    def ProcessWAR(self):
        ''' Taking into account that more than one reply can be received, it may be the case
        just some of them send a warning packet but not all, so, only those are discarded.
        '''
        self.logger.info("Ok, I won't consider you " + self.incoming_packet["source"])

        try:
            del self.valid_candidates[self.incoming_packet["source"]]
        except KeyError:
            self.logger.info("You were not in my list " + self.incoming_packet["source"] + ". Your packet may have collided.")

        if len(self.valid_candidates) == 0:
            if self.multicast == True and self.fsm.current_state == "WAIT_TIME":
                # All candidates have sent a WAR packet or the only one did
                self.fsm.process("defer")
            elif self.multicast == False and self.fsm.current_state == "WAIT_TIME":
                self.fsm.process("defer")

        self.incoming_packet = None


    def SendRTS(self):
        ''' The RTS sent is the normal one, but we should initialize the list of replies.
        '''
        self.valid_candidates = {}
        DACAP.SendRTS(self)


class CSMA:
    """CSMA: Carrier Sensing Multiple Access - Something between ALOHA and DACAP
    """

    def __init__(self, layercake, config):
        self.layercake = layercake
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)

        self.InitialiseStateEngine()
        self.timer = self.InternalTimer(self.fsm)
        self.TimerRequest = Sim.SimEvent("TimerRequest")

        self.outgoing_packet_queue = []
        self.incoming_packet = None
        self.last_packet = None

        self.packet_signal = {"ACK": "got_ACK", "RTS": "got_RTS", "CTS": "got_CTS", "DATA": "got_DATA"}
        self.ack_packet_length = config["ack_packet_length"]
        self.rts_packet_length = config["rts_packet_length"]
        self.cts_packet_length = config["cts_packet_length"]

        self.max_transmission_attempts = config["attempts"]

        self.transmission_attempts = 0
        self.max_wait_to_retransmit = config["max2resend"]
        self.channel_access_retries = 0
        self.next_timeout = 0
        self.pending_data_packet_from = None
        self.pending_ack_packet_from = None

        # Timing parameters
        self.T = 0  # It will be adapted according to the transmission range

        self.t_control = self.rts_packet_length / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)

    def activate(self):
        Sim.activate(self.timer, self.timer.Lifecycle(self.TimerRequest))


    class InternalTimer(Sim.Process):
        def __init__(self, fsm):
            Sim.Process.__init__(self, name="MAC_Timer")
            random.seed()
            self.fsm = fsm

        def Lifecycle(self, Request):
            while True:
                yield Sim.waitevent, self, Request
                yield Sim.hold, self, Request.signalparam[0]
                if (self.interrupted()):
                    self.interruptReset()
                else:
                    if not Request.occurred:
                        self.fsm.process(Request.signalparam[1])


    def InitialiseStateEngine(self):
        """InitialiseStateEngine: set up Finite State Machine for RTSCTS
        """
        self.fsm = FSM("READY_WAIT", [], self)

        #Set default to Error
        self.fsm.set_default_transition(self.OnError, "READY_WAIT")

        #Transitions from READY_WAIT
        # Normal transitions
        self.fsm.add_transition("send_DATA", "READY_WAIT", self.SendRTS, "READY_WAIT")  #Only if the channel is idle will I transmit
        self.fsm.add_transition("got_RTS", "READY_WAIT", self.CheckRTS, "READY_WAIT")  #Check if it is duplicated

        self.fsm.add_transition("got_X", "READY_WAIT", self.XOverheard, "BACKOFF")
        # Strange but possible transitions
        self.fsm.add_transition("got_DATA", "READY_WAIT", self.CheckPendingData, "READY_WAIT")
        self.fsm.add_transition("got_ACK", "READY_WAIT", self.CheckPendingACK, "READY_WAIT")

        #Transitions from WAIT_CTS
        # Normal transitions
        self.fsm.add_transition("send_DATA", "WAIT_CTS", self.QueueData, "WAIT_CTS")
        self.fsm.add_transition("got_CTS", "WAIT_CTS", self.Transmit, "WAIT_ACK")  #This is the main difference with DACAP
        self.fsm.add_transition("got_RTS", "WAIT_CTS", self.IgnoreRTS, "WAIT_CTS")
        self.fsm.add_transition("timeout", "WAIT_CTS", self.OnCTSTimeout, "READY_WAIT")

        self.fsm.add_transition("got_X", "WAIT_CTS", self.XOverheard, "WAIT_CTS")
        self.fsm.add_transition("backoff", "WAIT_CTS", self.XOverheard, "BACKOFF")
        # Strange but possible transitions
        self.fsm.add_transition("got_ACK", "WAIT_CTS", self.OnTransmitSuccess, "READY_WAIT")  #I was transmitting a duplicated packet

        #Transitions from WAIT_DATA
        self.fsm.add_transition("send_DATA", "WAIT_DATA", self.QueueData, "WAIT_DATA")
        self.fsm.add_transition("got_RTS", "WAIT_DATA", self.IgnoreRTS, "WAIT_DATA")
        self.fsm.add_transition("got_CTS", "WAIT_DATA", self.IgnoreCTS, "WAIT_DATA")
        self.fsm.add_transition("got_DATA", "WAIT_DATA", self.OnDataReception, "WAIT_DATA")
        self.fsm.add_transition("timeout", "WAIT_DATA", self.OnDATATimeout, "READY_WAIT")

        self.fsm.add_transition("got_X", "WAIT_DATA", self.XOverheard, "WAIT_DATA")
        self.fsm.add_transition("backoff", "WAIT_DATA", self.XOverheard, "BACKOFF")

        #Transitions from WAIT_ACK
        self.fsm.add_transition("send_DATA", "WAIT_ACK", self.QueueData, "WAIT_ACK")
        self.fsm.add_transition("got_ACK", "WAIT_ACK", self.OnTransmitSuccess, "READY_WAIT")
        self.fsm.add_transition("got_RTS", "WAIT_ACK", self.IgnoreRTS, "WAIT_ACK")
        self.fsm.add_transition("timeout", "WAIT_ACK", self.OnACKTimeout, "READY_WAIT")

        self.fsm.add_transition("got_X", "WAIT_ACK", self.XOverheard, "WAIT_ACK")
        self.fsm.add_transition("backoff", "WAIT_ACK", self.XOverheard, "BACKOFF")

        #Transitions from BACKOFF
        self.fsm.add_transition("send_DATA", "BACKOFF", self.QueueData, "BACKOFF")
        self.fsm.add_transition("got_RTS", "BACKOFF", self.ProcessRTS, "BACKOFF")
        ##self.fsm.add_transition("got_CTS", "BACKOFF", self.IgnoreCTS, "BACKOFF")  ### This line is more important that what it seems: if we Ignore it, we tend to defer all transmissions.
        self.fsm.add_transition("got_CTS", "BACKOFF", self.Transmit, "WAIT_ACK")    ### This line is more important that what it seems: if we Accept it, we tend to make all transmissions.
        self.fsm.add_transition("got_DATA", "BACKOFF", self.CheckPendingData, "BACKOFF")
        self.fsm.add_transition("got_ACK", "BACKOFF", self.CheckPendingACK, "READY_WAIT")  ### Be careful with this

        self.fsm.add_transition("got_X", "BACKOFF", self.XOverheard, "BACKOFF")
        self.fsm.add_transition("timeout", "BACKOFF", self.OnTimeout, "READY_WAIT")
        self.fsm.add_transition("accept", "BACKOFF", self.SendCTS, "WAIT_DATA")


    def XOverheard(self):
        ''' By overhearing the channel, we can obtain valuable information.
        '''
        if self.fsm.current_state == "READY_WAIT":
            if debug: self.logger.info("I'm in READY_WAIT and have received a " + self.incoming_packet["type"])

        elif self.fsm.current_state == "WAIT_CTS":
            if debug: self.logger.info("I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
            self.pending_ack_packet_from = self.last_data_to
            self.fsm.process("backoff")
            return False

        if self.fsm.current_state == "WAIT_DATA":
            if debug: self.logger.info("I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
            self.pending_data_packet_from = self.last_cts_to
            self.fsm.process("backoff")
            return False

        elif self.fsm.current_state == "WAIT_ACK":
            if debug: self.logger.info("I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
            self.pending_ack_packet_from = self.last_data_to
            self.fsm.process("backoff")
            return False

        self.last_packet = self.incoming_packet

        T = self.layercake.phy.level2delay(self.level)

        backoff = self.BackOff(self.incoming_packet["type"], T)

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.next_timeout > Sim.now() + backoff:
            # I will wait for the longest time
            backoff = self.next_timeout - Sim.now()

        if debug: self.logger.debug("Sleep for " + str(backoff) + " due to " + self.incoming_packet["type"]
                                    + " coming from " + self.incoming_packet["source"]
                                    + " to " + self.incoming_packet["dest"]
                                    + " through " + self.incoming_packet["through"])

        self.TimerRequest.signal((backoff, "timeout"))
        self.next_timeout = Sim.now() + backoff

        self.incoming_packet = None


    def IgnoreRTS(self):
        self.logger.debug("Ignoring RTS received from {src} as I'm currently in {state}".format(
            src=self.incoming_packet["source"],
            state=self.fsm.current_state))
        self.incoming_packet = None


    def IgnoreCTS(self):
        self.logger.debug("Ignoring CTS received from {src} as I'm waiting on an ACK".format(src=self.incoming_packet["source"]))
        self.incoming_packet = None


    def IgnoreACK(self):
        self.logger.debug("Ignoring ACK coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def CheckPendingData(self):
        if self.pending_data_packet_from != None:
            if self.incoming_packet["route"][-1][0] == self.pending_data_packet_from:
                self.logger.warn("Despite everything, I properly received the DATA packet from: " + self.incoming_packet["source"])
                self.OnDataReception()
                self.pending_data_packet_from = None
        else:
            self.logger.info("I think I have pending data from {src} but I don't".format(
                src=self.incoming_packet["source"])
            )

            self.OnError()


    def CheckPendingACK(self):
        if self.pending_ack_packet_from is not None:
            if self.incoming_packet["source"] == self.pending_ack_packet_from or self.pending_ack_packet_from[0:3] == "ANY":
                self.logger.warn("Despite everything, the DATA was properly transmitted to: " + self.incoming_packet["source"])
                self.OnTransmitSuccess()
                self.pending_ack_packet_from = None
        else:
            self.logger.info("I think I have pending ACK from {src} but I don't".format(
                src=self.incoming_packet["source"])
            )
            self.OnError()


    def SendRTS(self):
        ''' Request To Send.
        '''
        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2delay(self.level)

        if self.layercake.phy.IsIdle():
            self.transmission_attempts = self.transmission_attempts + 1
            self.channel_access_retries = 0

            if self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
                self.multicast = True
            else:
                self.multicast = False

            RTSPacket = {"type": "RTS", "ID": self.outgoing_packet_queue[0]["ID"],
                         "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                         "dest": self.outgoing_packet_queue[0]["dest"], "dest_position": self.outgoing_packet_queue[0]["dest_position"],
                         "through": self.outgoing_packet_queue[0]["through"], "through_position": self.outgoing_packet_queue[0]["through_position"],
                         "length": self.rts_packet_length, "level": self.outgoing_packet_queue[0]["level"], "time_stamp": Sim.now()}

            self.last_data_to = self.outgoing_packet_queue[0]["through"]
            self.level = self.outgoing_packet_queue[0]["level"]
            self.T = self.layercake.phy.level2delay(self.level)

            self.layercake.phy.TransmitPacket(RTSPacket)
            self.t_data = RTSPacket['length'] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)

            timeout = self.TimeOut("CTS", self.T)
            if debug: self.logger.debug("Transmitting RTS to " + self.outgoing_packet_queue[0]["dest"] + " through " + self.outgoing_packet_queue[0]["through"] + " with power level " + str(self.outgoing_packet_queue[0]["level"]) + " waiting " + str(timeout))
            self.TimerRequest.signal((timeout, "timeout"))
            self.fsm.current_state = "WAIT_CTS"

            self.time = Sim.now()
            self.next_timeout = self.time + timeout

        else:  #I'm currently not limiting the number of channel access retries
            self.channel_access_retries = self.channel_access_retries + 1
            self.t_data = self.outgoing_packet_queue[0]["length"] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
            timeout = random.random() * (2 * self.T + self.t_data)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))

            self.time = Sim.now()
            self.next_timeout = self.time + timeout


    def ProcessRTS(self):
        ''' Maybe I can do it now.
        '''
        if self.last_packet["type"] == "CTS" and self.last_packet["through"] == self.incoming_packet["source"]:
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")
        elif self.last_packet["type"] == "DATA" and self.last_packet["through"] == self.incoming_packet["source"] and self.last_packet["dest"] == self.incoming_packet["dest"]:
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")


    def BackOff(self, packet_type, T):
        ''' Returns the backoff for a specific state.
        '''
        if packet_type == "RTS":
            backoff = 4 * T
        elif packet_type == "CTS" or packet_type == "SIL":
            backoff = 3 * T
        elif packet_type == "DATA":
            backoff = 2 * T
        elif packet_type == "ACK":
            backoff = 0  #I'm all set

        backoff = np.random.random() * (backoff)
        return backoff


    def TimeOut(self, packet_type, T):
        ''' Returns the timeout for a specific state.
        '''

        if packet_type == "CTS":
            t = 2 * T + 2 * self.t_control
        elif packet_type == "DATA":
            t = 2 * T + self.t_data + self.t_control
        elif packet_type == "ACK":
            t = 2 * T + self.t_data + self.t_control

        return np.random.random() * (t)


    def ProcessCTS(self):
        ''' The CTS has been received.
        '''
        self.logger.info("CTS from " + self.incoming_packet["source"] + " coming from " + self.incoming_packet["through"] + " properly received.")

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        self.incoming_packet = None


    def CheckRTS(self):
        ''' Before proceeding with the data transmission, just check if it's a duplicated packet (maybe the ACK collided).
        '''
        if self.layercake.net.IsDuplicated(self.incoming_packet):
            self.SendACK(self.incoming_packet["source"])
        else:
            self.SendCTS()
            self.fsm.current_state = "WAIT_DATA"


    def SendCTS(self):
        ''' Clear To Send: I'm proposing myself as a good candidate for the next transmission or I just let transmit if I have been
        already selected.
        '''
        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        CTSPacket = {"type": "CTS", "ID": self.incoming_packet["ID"],
                     "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                     "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                     "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                     "length": self.cts_packet_length, "rx_energy": self.layercake.phy.rx_energy,
                     "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

        if debug: self.logger.debug("Transmitting CTS to " + self.incoming_packet["source"])
        self.last_cts_to = self.incoming_packet["source"]
        self.last_cts_from = self.incoming_packet["dest"]

        self.layercake.phy.TransmitPacket(CTSPacket)
        self.t_data = CTSPacket['length'] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)

        self.level = self.incoming_packet["level"]
        self.T = self.layercake.phy.level2delay(self.level)

        timeout = self.TimeOut("DATA", self.T)
        self.TimerRequest.signal((timeout, "timeout"))

        self.time = Sim.now()
        self.next_timeout = self.time + timeout

        self.incoming_packet = None


    def SendACK(self, packet_origin):
        ''' Sometimes we can not use implicit ACKs.
        '''
        if debug: self.logger.debug("ACK to " + packet_origin)

        AckPacket = {"type": "ACK",
                     "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                     "dest": packet_origin, "dest_position": None,
                     "through": packet_origin, "through_position": None,
                     "length": self.ack_packet_length, "level": self.incoming_packet["level"],
                     "ID": self.incoming_packet["ID"]}

        self.layercake.phy.TransmitPacket(AckPacket)


    def InitiateTransmission(self, OutgoingPacket):
        ''' Function called from the upper layers to transmit a packet.
        '''
        if debug: self.logger.debug("Prepping {id} for launch to {dest}".format(
            id=OutgoingPacket["ID"],
            dest=OutgoingPacket["dest"]
        )
        )
        self.outgoing_packet_queue.append(OutgoingPacket)
        self.fsm.process("send_DATA")


    def OnNewPacket(self, IncomingPacket):
        ''' Function called from the lower layers when a packet is received.
        '''
        self.incoming_packet = IncomingPacket
        if self.CanIHelp(IncomingPacket):
            self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            self.OverHearing()


    def CanIHelp(self, packet):
        ''' A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        '''
        # Is this a packet already "directed" to me?
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTS and ACK are the only types of packet that are directly addressed
        if packet["type"] == "CTS" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "ACK" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["dest"] == broadcast_address:
            return True

        return False


    def OverHearing(self):
        ''' Valuable information can be obtained from overhearing the channel.
        '''
        if self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet["dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")
        else:
            self.fsm.process("got_X")


    def OnDataReception(self):
        ''' After the RTS/CTS exchange, a data packet is received. We should acknowledge the previous node or we can try to use implicit
        acknowledges if the that does not mean a waste of power.
        '''
        if self.fsm.current_state != "READY_WAIT":
            # We got a DATA packet, we should stop the timer.
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.current_state = "READY_WAIT"

        if self.fsm.current_state != "BACKOFF":
            self.last_packet = None

        # We should acknowledge the previous node, this is not an end-to-end ACK
        packet_origin = self.incoming_packet["route"][-1][0]

        if self.layercake.net.NeedExplicitACK(self.incoming_packet["level"], self.incoming_packet["dest"]) or len(self.outgoing_packet_queue) != 0:
            self.SendACK(packet_origin)

        self.layercake.net.OnPacketReception(self.incoming_packet)

        self.incoming_packet = None


    def OnError(self):
        ''' This function is called when the FSM has an unexpected input_symbol in a determined state.
        '''
        try:
            self.logger.error("Unexpected transition from {sm.last_state} to {sm.current_state} due to {sm.input_symbol}: {src} {dest} {id}".format(
                sm=self.fsm,
                src=self.incoming_packet['source'],
                dest=self.incoming_packet['dest'],
                id=self.incoming_packet['ID']
                )
            )
        except:
            raise RuntimeError("REALLY Unexpected transition from {sm.last_state} to {sm.current_state} due to {sm.input_symbol}: {pkt}".format(
                sm=self.fsm,
                pkt=self.incoming_packet
            )
            )

    def OnTransmitSuccess(self):
        ''' When an ACK is received, we can assume that everything has gone fine, so it's all done.
        '''
        if debug: self.logger.info("Successfully Transmitted to " + self.incoming_packet["source"])

        # We got an ACK, we should stop the timer.
        p = Sim.Process()
        p.interrupt(self.timer)

        self.PostSuccessOrFail()


    def OnTransmitFail(self):
        ''' All the transmission attemps have been completed. It's impossible to reach the node.
        '''
        self.logger.info("Failed to transmit to " + self.outgoing_packet_queue[0]["dest"])
        self.PostSuccessOrFail()


    def PostSuccessOrFail(self):
        ''' Successfully or not, we have finished the current transmission.
        '''
        self.outgoing_packet_queue.pop(0)["dest"]
        self.transmission_attempts = 0

        # Is there anything else to do?
        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0


    def OnACKTimeout(self):
        ''' The timeout has experied and NO ACK has been received.
        '''
        self.transmission_attempts += 1

        if self.transmission_attempts < self.max_transmission_attempts:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
        else:
            self.logger.debug("Timed Out after {}: No Ack".format(self.transmission_attempts))


    def OnDATATimeout(self):
        ''' The timeout has experied and NO DATA has been received.
        '''
        self.logger.info("Timed Out!, No Data Received")

        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0


    def OnCTSTimeout(self):
        self.transmission_attempts += 1
        self.logger.debug("CTS timeout waiting on {}: Attempt {}".format(
            self.outgoing_packet_queue[0]['ID'],
            self.transmission_attempts)
        )
        if self.layercake.phy.CollisionDetected():
            self.logger.info("It seems that there has been a collision.")

        if self.transmission_attempts > self.max_transmission_attempts:
            self.logger.info("CTS timeout limit waiting on {} after {}".format(
                self.outgoing_packet_queue[0]['ID'],
                self.transmission_attempts)
            )
        else:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))


    def OnTimeout(self):
        if debug: self.logger.debug("Exiting from back off")

        if (len(self.outgoing_packet_queue) > 0):
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0


    def Transmit(self):
        p = Sim.Process()
        p.interrupt(self.timer)

        self.TransmitNoTimer();


    def TransmitNoTimer(self):
        self.layercake.phy.TransmitPacket(self.outgoing_packet_queue[0])
        self.last_data_to = self.outgoing_packet_queue[0]["through"]
        timeout = self.TimeOut("ACK", self.T)
        self.TimerRequest.signal((timeout, "timeout"))


    def QueueData(self):
        self.logger.info("Queuing Data")


class CSMA4FBR(CSMA):
    '''CSMA for FBR: adaptation of CSMA with power control to couple it with FBR
    '''

    def __init__(self, node, config):

        CSMA.__init__(self, node, config)

        # A new packet type
        self.packet_signal["SIL"] = "got_SIL"
        self.SIL_packet_length = config["SIL_packet_length"]

        self.valid_candidates = {}
        self.original_through = None

        # By MC_RTS we refer to MultiCast RTS packets. We should consider to process them.
        self.fsm.add_transition("got_MC_RTS", "READY_WAIT", self.ProcessMCRTS, "AWARE")
        self.fsm.add_transition("got_SIL", "READY_WAIT", self.IgnoreSIL, "READY_WAIT")

        # Definition of a new state, AWARE, and its transitions
        self.fsm.add_transition("send_CTS", "AWARE", self.SendCTS, "WAIT_DATA")
        self.fsm.add_transition("ignore_RTS", "AWARE", self.XOverheard, "BACKOFF")

        # Now I am receiving several CTS, then, I should just append them
        self.fsm.add_transition("got_MC_RTS", "WAIT_CTS", self.ProcessMCRTS, "WAIT_CTS")  # Maybe I should tell him to be silent
        self.fsm.add_transition("got_CTS", "WAIT_CTS", self.AppendCTS, "WAIT_CTS")
        self.fsm.add_transition("got_SIL", "WAIT_CTS", self.ProcessSIL, "BACKOFF")

        self.fsm.add_transition("timeout", "WAIT_CTS", self.SelectCTS, "WAIT_CTS")
        self.fsm.add_transition("transmit", "WAIT_CTS", self.Transmit, "WAIT_ACK")
        self.fsm.add_transition("retransmit", "WAIT_CTS", self.SendRTS, "WAIT_CTS")
        self.fsm.add_transition("abort", "WAIT_CTS", self.OnTransmitFail, "READY_WAIT")
        self.fsm.add_transition("defer", "WAIT_CTS", self.XOverheard, "BACKOFF")

        # New transition from WAIT_DATA: I have been not selected as the next hop
        self.fsm.add_transition("got_MC_RTS", "WAIT_DATA", self.ProcessMCRTS, "WAIT_DATA")
        self.fsm.add_transition("ignored", "WAIT_DATA", self.XOverheard, "BACKOFF")
        self.fsm.add_transition("got_CTS", "WAIT_DATA", self.IgnoreCTS, "WAIT_DATA")
        self.fsm.add_transition("got_SIL", "WAIT_DATA", self.IgnoreSIL, "WAIT_DATA")

        self.fsm.add_transition("got_MC_RTS", "BACKOFF", self.ProcessMCRTS, "BACKOFF")
        self.fsm.add_transition("got_SIL", "BACKOFF", self.IgnoreSIL, "BACKOFF")
        self.fsm.add_transition("defer", "BACKOFF", self.XOverheard, "BACKOFF")

        self.fsm.add_transition("got_MC_RTS", "WAIT_ACK", self.ProcessMCRTS, "WAIT_ACK")


    def IgnoreSIL(self):
        self.logger.info("Ignoring SIL coming from " + self.incoming_packet["through"])
        self.incoming_packet = None


    def ProcessMCRTS(self):
        ''' Someone is looking for help, may I help? Now the active nodes are the ones that transmit, maybe we should do it the opposite way
        '''
        if self.fsm.current_state == "AWARE":
            if self.layercake.net.ImAValidCandidate(self.incoming_packet):
                if self.layercake.net.IsDuplicated(self.incoming_packet):
                    self.SendACK(self.incoming_packet["source"])
                    self.fsm.current_state = "READY_WAIT"
                else:
                    self.multicast = True
                    self.fsm.process("send_CTS")
            else:
                self.logger.info("I can't attend the MultiCast RTS received from " + self.incoming_packet["source"] + " but I will be silent.")
                self.fsm.process("ignore_RTS")
        elif self.fsm.current_state == "WAIT_DATA":
            if self.last_cts_to == self.incoming_packet["source"]:
                self.IgnoreRTS()  # Think of this
                # The silence part is being revised.
                ##            else:
                ##                self.SendSilence()
                ##        elif self.fsm.current_state == "WAIT_CTS" or self.fsm.current_state == "WAIT_TIME":
                ##            self.SendSilence()
                ##        elif self.fsm.current_state == "BACKOFF":
                ##            self.SendSilence()


    def SendSilence(self):
        ''' Please be quiet!
        '''
        SILPacket = {"type": "SIL",
                     "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                     "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                     "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                     "length": self.SIL_packet_length, "tx_energy": self.layercake.phy.tx_energy,
                     "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

        self.logger.info("Transmitting SIL to " + self.incoming_packet["source"])

        self.layercake.phy.TransmitPacket(SILPacket)

        self.incoming_packet = None


    def ProcessSIL(self):
        ''' The next time that I try to find a route I should do it starting again from ANY0.
        '''
        self.outgoing_packet_queue[0]["through"] = "ANY0"
        self.outgoing_packet_queue[0]["level"] = 0
        self.XOverheard()


    def AppendCTS(self):
        ''' More than one CTS is received when looking for the next best hop. We should consider all of them.
        The routing layer decides.
        '''
        self.valid_candidates[self.incoming_packet["through"]] = (Sim.now() - self.incoming_packet["time_stamp"]) / 2.0, self.incoming_packet["rx_energy"], self.incoming_packet["through_position"]
        self.logger.info("Appending CTS to " + self.incoming_packet["source"] + " coming from " + self.incoming_packet["through"])
        self.layercake.net.AddNode(self.incoming_packet["through"], self.incoming_packet["through_position"])

        self.incoming_packet = None

        if self.multicast == False:
            # Update the timer: 1.-Stop, 2.-Restart
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("timeout")


    def SelectCTS(self):
        ''' Once we have wait enough, that is, 2 times the distance at which the best next hop should be, we should select it.
        '''
        current_through = self.outgoing_packet_queue[0]["through"]
        self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0]["through_position"] = self.layercake.net.SelectRoute(self.valid_candidates, self.outgoing_packet_queue[0]["through"], self.transmission_attempts, self.outgoing_packet_queue[0]["dest"])

        if self.outgoing_packet_queue[0]["through"] == "ABORT":
            # We have consumed all the attemps
            self.fsm.process("abort")

        elif self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
            # We should retransmit increasing the power
            self.outgoing_packet_queue[0]["level"] = int(self.outgoing_packet_queue[0]["through"][3])
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "2CHANCE":
            # We should give a second chance to current node
            self.outgoing_packet_queue[0]["through"] = current_through
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "NEIGH":
            # With current transmission power level, the destination has become a neighbor
            self.outgoing_packet_queue[0]["through"] = self.outgoing_packet_queue[0]["dest"]
            self.outgoing_packet_queue[0]["through_position"] = self.outgoing_packet_queue[0]["dest_position"]
            self.fsm.process("retransmit")

        else:
            self.fsm.process("transmit")


    def Transmit(self):
        ''' Real Transmission of the Packet.
        '''
        self.layercake.net.Update(self.outgoing_packet_queue[0]["dest"], self.outgoing_packet_queue[0]["dest_position"], self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0]["through_position"])
        CSMA.TransmitNoTimer(self)


    def TimeOut(self, packet_type, T):
        ''' Returns the timeout for a specific state.
        '''

        if packet_type == "CTS":
            t = 2 * T + 2 * self.t_control
        elif packet_type == "DATA":
            if self.multicast:
                t = 3 * T + self.t_data + self.t_control
            elif not self.multicast:
                t = 2 * T + self.t_data + self.t_control
        elif packet_type == "ACK":
            t = 2 * T + self.t_data + self.t_control

        t = np.random.random() * (t)
        return t

    def OverHearing(self):
        ''' Valuable information can be obtained from overhearing the channel.
        '''
        if self.incoming_packet["type"] == "DATA" and self.fsm.current_state == "WAIT_DATA":
            if self.incoming_packet["route"][-1][0] == self.last_cts_to and self.incoming_packet["dest"] == self.last_cts_from:
                # I have not been selected as the next hop. I read it from the data.
                self.fsm.process("ignored")

        elif self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet["dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")

        elif self.incoming_packet["type"] == "CTS" and self.fsm.current_state == "WAIT_DATA" and self.last_cts_to == self.incoming_packet["dest"]:
            # This is another candidate proposing himself as a good candidate
            self.fsm.process("got_CTS")

        elif self.incoming_packet["type"] == "SIL":
            return False

        else:
            self.fsm.process("got_X")


    def CanIHelp(self, packet):
        ''' A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        '''
        # Is this a packet already "directed" to me? I'm not checking if only dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTS, ACK, WAR and SIL are the only types of packet that are directly addressed
        if packet["type"] == "CTS" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "ACK" and packet["dest"] == self.layercake.hostname:
            return True
        if packet["type"] == "SIL" and packet["dest"] == self.layercake.hostname:
            return True

        # Is this a multicast packet?
        if packet["through"][0:3] == "ANY":
            return True

        return False


    def OnNewPacket(self, IncomingPacket):
        ''' Function called from the lower layers when a packet is received.
        '''
        self.incoming_packet = IncomingPacket

        if self.CanIHelp(self.incoming_packet):
            if self.incoming_packet["through"][0:3] == "ANY":
                self.fsm.process("got_MC_RTS")
            else:
                self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            self.OverHearing()


    def SendRTS(self):
        ''' The RTS sent is the normal one, but we should initialize the list of replies.
        '''
        self.valid_candidates = {}
        CSMA.SendRTS(self)


    def PrintMessage(self, msg):
        pass
        #print "CSMA4FBR (%s): %s at t=" % (self.layercake.hostname, msg), Sim.now(), self.fsm.input_symbol, self.fsm.current_state
