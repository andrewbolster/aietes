# coding=utf-8
###########################################################################
#
# Copyright (C) 2007 by Justin Eskesen and Josep Miquel Jornet Montana
# <jge@mit.edu> <jmjornet@mit.edu>
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

import random

random.seed(123456789)

import logging

import SimPy.Simulation as Sim

from aietes.Layercake import RoutingLayer
from aietes.Tools import broadcast_address, DEBUG, distance
from aietes.Tools.FSM import FSM


# DEBUG = True

DEFAULT_PROTO = "ALOHA"


def setup_mac(node, config):
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


class MAC(object):
    total_channel_access_attempts = 0
    channel_access_retries = 0


class ALOHA(MAC):
    """ALOHA:  A very simple MAC Algorithm
    """

    def __init__(self, layercake, config):
        self.layercake = layercake
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)

        self.ack_packet_length = config["ack_packet_length"]
        self.packet_signal = {"ACK": "got_ACK", "DATA": "got_DATA"}

        self.initialise_state_machine()

        # Number of times that the channel was sensed and found not idle
        self.channel_access_retries = 0
        self.total_channel_access_attempts = 0
        self.transmission_attempts = 0  # Number of retransmissions
        self.max_transmission_attempts = config["attempts"]
        self.max_wait_to_retransmit = config["max2resend"]

        self.timeout = 0  # It is adapted with the transmission power level
        self.level = 0  # It will be defined from the routing layer
        self.T = 0  # It will be adapted once the level is fixed

        self.outgoing_packet_queue = []
        self.incoming_packet = None

        self.data_packet_length = config["data_packet_length"]

        self.stats = {"data_packets_sent": 0, "data_packets_received": {}}

        self.timer = self.InternalTimer(self.fsm)
        self.TimerRequest = Sim.SimEvent("TimerRequest")

    def activate(self):
        Sim.activate(self.timer, self.timer.lifecycle(self.TimerRequest))

    class InternalTimer(Sim.Process):

        def __init__(self, fsm):
            Sim.Process.__init__(self, name="MAC_Timer")
            random.seed()
            self.fsm = fsm

        def lifecycle(self, request):
            while True:
                yield Sim.waitevent, self, request
                yield Sim.hold, self, request.signalparam[0]
                if self.interrupted():
                    self.interruptReset()
                else:
                    self.fsm.process(request.signalparam[1])

    def initialise_state_machine(self):
        """initialise_state_machine:  set up Finite State Machine for ALOHA
        """
        self.fsm = FSM("READY_WAIT", [], self)

        # Set default to Error
        self.fsm.set_default_transition(self.on_error, "READY_WAIT")

        # Transitions from READY_WAIT
        self.fsm.add_transition(
            "got_DATA", "READY_WAIT", self.on_data_reception, "READY_WAIT")
        self.fsm.add_transition(
            "send_DATA", "READY_WAIT", self.transmit, "READY_WAIT")

        # Transitions from WAIT_ACK
        self.fsm.add_transition(
            "got_DATA", "WAIT_ACK", self.on_data_reception, "WAIT_ACK")
        self.fsm.add_transition(
            "send_DATA", "WAIT_ACK", self.queue_data, "WAIT_ACK")
        self.fsm.add_transition(
            "got_ACK", "WAIT_ACK", self.on_transmit_success, "READY_WAIT")
        self.fsm.add_transition(
            "timeout", "WAIT_ACK", self.on_timeout, "WAIT_2_RESEND")

        # Transitions from WAIT_2_RESEND
        self.fsm.add_transition(
            "resend", "WAIT_2_RESEND", self.transmit, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "got_DATA", "WAIT_2_RESEND", self.on_data_reception, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "send_DATA", "WAIT_2_RESEND", self.queue_data, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "fail", "WAIT_2_RESEND", self.on_transmit_fail, "READY_WAIT")

    def initiate_transmission(self, outgoing_packet):
        """ Function called from the upper layers to transmit a packet.
        """
        self.outgoing_packet_queue.append(outgoing_packet)
        self.fsm.process("send_DATA")

    def on_new_packet_received(self, incoming_packet):
        """ Function called from the lower layers when a packet is received.
        """
        self.incoming_packet = incoming_packet
        if self.is_for_me():
            if DEBUG:
                self.logger.debug(
                    "received Packet from {src}".format(src=incoming_packet['source']))
            self.fsm.process(self.packet_signal[incoming_packet["type"]])
        else:
            if DEBUG:
                self.logger.debug("Overheard Packet from {src} to {dest}".format(
                    src=incoming_packet['source'], dest=incoming_packet['dest']))
            self.overhearing()

    def is_for_me(self):
        """


        :return:
        """
        if self.layercake.hostname == self.incoming_packet["through"]:
            return True
        else:
            return broadcast_address == self.incoming_packet["through"]

    def overhearing(self):
        """ Valuable information can be obtained from overhearing the channel.
        """
        if self.incoming_packet["type"] == "DATA" and self.fsm.current_state == "WAIT_ACK":
            packet_origin = self.incoming_packet["route"][-1][0]
            if packet_origin == self.outgoing_packet_queue[0]["through"] and self.incoming_packet["dest"] == \
                    self.outgoing_packet_queue[0]["dest"]:
                if self.incoming_packet["ID"] == self.outgoing_packet_queue[0]["ID"]:
                    # This is an implicit ACK
                    self.logger.debug("An implicit ACK has arrived.")
                    self.fsm.process("got_ACK")

    def on_data_reception(self):
        """ A data packet is received. We should acknowledge the previous node or we can try to use implicit
            acknowledges if that does not mean a waste of power.
        """
        packet_origin = self.incoming_packet[
            "route"][-1][0]  # These ACKs just gone to the previous hop, this is maintenance at MAC layer

        if self.layercake.net.need_explicit_ack(self.incoming_packet["level"], self.incoming_packet["dest"]) \
                or len(self.outgoing_packet_queue) != 0 \
                or self.layercake.net.have_duplicate_packet(self.incoming_packet):
            self.send_ack(packet_origin)

        self.layercake.net.on_packet_reception(self.incoming_packet)

    def send_ack(self, packet_origin):
        """

        :param packet_origin:
        """
        if DEBUG:
            self.logger.debug("ACK to " + packet_origin)
        ack_packet = {"type": "ACK",
                      "source": self.layercake.hostname,
                      "source_position": self.layercake.get_current_position(),
                      "dest": packet_origin,
                      "dest_position": self.incoming_packet["source_position"],
                      "through": packet_origin,
                      "through_position": self.incoming_packet["source_position"],
                      "length": self.ack_packet_length,
                      "level": self.incoming_packet["level"],
                      "ID": self.incoming_packet["ID"]}

        self.layercake.phy.transmit_packet(ack_packet)

    def on_error(self):
        """ This function is called when the FSM has an unexpected input_symbol in a determined state.
        """
        self.logger.error(
            "Unexpected transition from {sm.last_state} to {sm.current_state} due to {sm.input_symbol}: {pkt}".format(
                sm=self.fsm, pkt=self.incoming_packet))

    def on_transmit_success(self):
        """ When an ACK is received, we can assume that everything has gone fine, so it's all done.
        """
        if DEBUG:
            self.logger.debug(
                "Successfully Transmitted to " + self.incoming_packet["source"])

        # We got an ACK, stop the timer...
        p = Sim.Process()
        p.interrupt(self.timer)

        self.post_success_or_fail()

    def on_transmit_fail(self):
        """ All the transmission attempts have been completed. It's impossible to reach the node.
        """
        self.logger.error("Failed to transmit to {through} with id {id}".format(
            through=self.outgoing_packet_queue[0]["through"],
            id=self.outgoing_packet_queue[0]["ID"])
        )
        self.post_success_or_fail()

    def post_success_or_fail(self):
        """ Successfully or not, we have finished the current transmission.
        """
        self.outgoing_packet_queue.pop(0)
        self.transmission_attempts = 0
        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))

    def on_timeout(self):
        """


        """
        self.transmission_attempts += 1
        if DEBUG:
            self.logger.debug("Timed Out, No Ack Received")

        if self.transmission_attempts > self.max_transmission_attempts:
            self.fsm.process("fail")
        else:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "resend"))

    def transmit(self):
        """ Real Transmission of the Packet.
        """
        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2delay(self.level)
        self.t_data = self.outgoing_packet_queue[0][
                          "length"] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
        self.timeout = 2 * self.T + 2 * self.t_data

        # Before transmitting, we check if the channel is idle
        if self.layercake.phy.is_idle():
            self.total_channel_access_attempts += 1
            self.transmission_attempts += 1
            self.channel_access_retries = 0
            if DEBUG:
                self.logger.debug(
                    "transmit to " + self.outgoing_packet_queue[0]["through"])
            self.layercake.phy.transmit_packet(self.outgoing_packet_queue[0])
            self.fsm.current_state = "WAIT_ACK"
            self.TimerRequest.signal((self.timeout, "timeout"))
            if DEBUG:
                self.logger.debug("The timeout is " + str(self.timeout))
        else:
            self.channel_access_retries += 1
            timeout = random.random() * (2 * self.T)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))

    def queue_data(self):
        """


        """
        self.logger.debug("Queuing Data")

    def print_msg(self, msg):
        """

        :param msg:
        """
        pass
        # print "ALOHA (%s): %s at t=" % (self.layercake.hostname, msg),
        # Sim.now(), self.fsm.input_symbol, self.fsm.current_state


class ALOHA4FBR(ALOHA):
    """ CS-ALOHA adapted for the FBR protocol.
    """

    def __init__(self, node, config):

        ALOHA.__init__(self, node, config)

        # New packet types
        self.packet_signal["RTS"] = "got_RTS"  # Route Request packet
        self.packet_signal["CTS"] = "got_CTS"  # Route Proposal packet

        self.rts_packet_length = config["rts_packet_length"]
        self.cts_packet_length = config["cts_packet_length"]

        self.valid_candidates = {}
        self.original_through = None

        # New Transitions from READY_WAIT - RTS packets are used to retrieve a
        # route
        self.fsm.add_transition(
            "send_DATA", "READY_WAIT", self.route_check, "READY_WAIT")
        self.fsm.add_transition(
            "send_RTS", "READY_WAIT", self.send_rts, "READY_WAIT")
        self.fsm.add_transition(
            "transmit", "READY_WAIT", self.transmit, "READY_WAIT")

        self.fsm.add_transition(
            "got_RTS", "READY_WAIT", self.process_rts, "READY_WAIT")
        self.fsm.add_transition(
            "send_CTS", "READY_WAIT", self.send_cts, "READY_WAIT")
        self.fsm.add_transition(
            "ignore_RTS", "READY_WAIT", self.ignore_rts, "READY_WAIT")

        self.fsm.add_transition(
            "got_ACK", "READY_WAIT", self.ignore_ack, "READY_WAIT")

        # New State in which we wait for the routes
        self.fsm.add_transition(
            "got_DATA", "WAIT_CTS", self.on_data_reception, "WAIT_CTS")
        self.fsm.add_transition(
            "send_DATA", "WAIT_CTS", self.queue_data, "WAIT_CTS")

        self.fsm.add_transition(
            "got_RTS", "WAIT_CTS", self.ignore_rts, "WAIT_CTS")

        self.fsm.add_transition(
            "got_CTS", "WAIT_CTS", self.append_cts, "WAIT_CTS")
        self.fsm.add_transition(
            "timeout", "WAIT_CTS", self.select_cts, "WAIT_CTS")
        self.fsm.add_transition(
            "transmit", "WAIT_CTS", self.transmit, "WAIT_CTS")
        self.fsm.add_transition(
            "retransmit", "WAIT_CTS", self.send_rts, "WAIT_CTS")
        self.fsm.add_transition(
            "send_RTS", "WAIT_CTS", self.send_rts, "WAIT_CTS")

        self.fsm.add_transition(
            "abort", "WAIT_CTS", self.on_transmit_fail, "READY_WAIT")
        self.fsm.add_transition(
            "got_ACK", "WAIT_CTS", self.on_transmit_success, "WAIT_CTS")

        # New Transitions from WAIT_ACK
        self.fsm.add_transition(
            "got_RTS", "WAIT_ACK", self.ignore_rts, "WAIT_ACK")
        self.fsm.add_transition(
            "got_CTS", "WAIT_ACK", self.ignore_cts, "WAIT_ACK")

        # New Transitions from WAIT_2_RESEND
        self.fsm.add_transition(
            "resend", "WAIT_2_RESEND", self.route_check, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "send_RTS", "WAIT_2_RESEND", self.send_rts, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "transmit", "WAIT_2_RESEND", self.transmit, "WAIT_2_RESEND")

        self.fsm.add_transition(
            "got_RTS", "WAIT_2_RESEND", self.process_rts, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "send_CTS", "WAIT_2_RESEND", self.send_cts, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "ignore_RTS", "WAIT_2_RESEND", self.ignore_rts, "WAIT_2_RESEND")
        self.fsm.add_transition(
            "got_ACK", "WAIT_2_RESEND", self.on_transmit_success, "WAIT_2_RESEND")

    def ignore_rts(self):
        """


        """
        self.logger.debug(
            "Ignoring RTS received from " + self.incoming_packet["source"])
        self.incoming_packet = None

    def ignore_cts(self):
        """


        """
        self.logger.debug(
            "Ignoring CTS coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def ignore_ack(self):
        """


        """
        self.logger.debug(
            "Ignoring ACK coming from " + self.incoming_packet["through"])
        print self.layercake.hostname, "Have we had a collision?"
        self.incoming_packet = None

    def send_rts(self):
        """ The RTS sent is the normal one, but we should initialize the list of replies.
        """
        self.valid_candidates = {}

        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2delay(self.level)

        if self.layercake.phy.is_idle():
            self.total_channel_access_attempts += 1
            self.transmission_attempts += 1
            self.channel_access_retries = 0

            if self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
                self.multicast = True
            else:
                self.multicast = False

            rts_packet = {"type": "RTS", "ID": self.outgoing_packet_queue[0]["ID"],
                          "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                          "dest": self.outgoing_packet_queue[0]["dest"],
                          "dest_position": self.outgoing_packet_queue[0]["dest_position"],
                          "through": self.outgoing_packet_queue[0]["through"],
                          "through_position": self.outgoing_packet_queue[0]["through_position"],
                          "length": self.rts_packet_length, "level": self.outgoing_packet_queue[0]["level"],
                          "time_stamp": Sim.now()}

            if DEBUG:
                self.logger.debug("Transmitting RTS to " + self.outgoing_packet_queue[0]["dest"] + " through " +
                                  self.outgoing_packet_queue[
                                      0]["through"] + " with power level " + str(
                    self.outgoing_packet_queue[0]["level"]))

            self.layercake.phy.transmit_packet(rts_packet)

            self.level = self.outgoing_packet_queue[0]["level"]
            self.T = self.layercake.phy.level2delay(self.level)
            timeout = 2 * self.T

            self.TimerRequest.signal((timeout, "timeout"))
            self.fsm.current_state = "WAIT_CTS"

        # I'm currently not limiting the number of channel access retries
        else:
            self.logger.debug("The channel was not idle.")
            self.channel_access_retries += 1
            timeout = random.random() * (2 * self.T + self.t_data)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))

    def process_rts(self):
        """ Someone is looking for help, may I help? Now the active nodes are the ones that transmit, maybe we should do it the opposite way
        """
        if self.fsm.current_state != "WAIT_ACK":
            if self.layercake.net.i_am_a_valid_candidate(self.incoming_packet):
                if self.layercake.net.have_duplicate_packet(self.incoming_packet):
                    self.SendACK(self.incoming_packet["source"])
                    self.fsm.current_state = "READY_WAIT"
                else:
                    self.multicast = True
                    self.fsm.process("send_CTS")
            else:
                self.logger.debug(
                    "I can't attend the RTS received from " + self.incoming_packet["source"] + ".")
                self.fsm.process("ignore_RTS")

    def route_check(self):
        """


        """
        if self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
            self.fsm.process("send_RTS")
        else:
            self.fsm.process("transmit")

    def send_cts(self):
        """ Clear To Send: I'm proposing myself as a good candidate for the next transmission or I just let transmit if I have been
        already selected.
        """
        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.layercake.phy.is_idle():
            cts_packet = {"type": "CTS",
                          "source": self.incoming_packet["dest"],
                          "source_position": self.incoming_packet["dest_position"],
                          "dest": self.incoming_packet["source"],
                          "dest_position": self.incoming_packet["source_position"],
                          "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                          "length": self.cts_packet_length, "rx_energy": self.layercake.phy.rx_energy,
                          "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

            if DEBUG:
                self.logger.debug(
                    "Transmitting CTS to " + self.incoming_packet["source"])
            self.layercake.phy.transmit_packet(cts_packet)

        self.incoming_packet = None

    def append_cts(self):
        """ More than one CTS is received when looking for the next best hop. We should consider all of them.
        The routing layer decides.
        """
        self.valid_candidates[self.incoming_packet["through"]] = (
                                                                     Sim.now() - self.incoming_packet[
                                                                         "time_stamp"]) / 2.0, self.incoming_packet[
                                                                     "rx_energy"], self.incoming_packet[
                                                                     "through_position"]
        self.logger.debug("Appending CTS to " + self.incoming_packet[
            "source"] + " coming from " + self.incoming_packet["through"])
        self.layercake.net.add_node(
            self.incoming_packet["through"], self.incoming_packet["through_position"])

        self.incoming_packet = None

        if not self.multicast:
            # Update the timer: 1.-Stop, 2.-Restart
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("timeout")

    def select_cts(self):
        """ Once we have wait enough, that is, 2 times the distance at which the best next hop should be, we should select it.
        """

        current_through = self.outgoing_packet_queue[0]["through"]
        self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0][
            "through_position"] = self.layercake.net.select_route(
            self.valid_candidates, self.outgoing_packet_queue[0]["through"], self.transmission_attempts,
            self.outgoing_packet_queue[0]["dest"])

        if self.outgoing_packet_queue[0]["through"] == "ABORT":
            # We have consumed all the attemps
            self.fsm.process("abort")

        elif self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
            # We should retransmit increasing the power
            self.outgoing_packet_queue[0]["level"] = int(
                self.outgoing_packet_queue[0]["through"][3])
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "2CHANCE":
            # We should give a second chance to current node
            self.outgoing_packet_queue[0]["through"] = current_through
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"][0:5] == "NEIGH":
            # With current transmission power level, the destination has become
            # a neighbor
            self.outgoing_packet_queue[0][
                "through"] = self.outgoing_packet_queue[0]["dest"]
            self.outgoing_packet_queue[0][
                "through_position"] = self.outgoing_packet_queue[0]["dest_position"]
            self.fsm.process("retransmit")

        else:
            self.fsm.process("transmit")

    def transmit(self):
        """ Real Transmission of the Packet.
        """
        self.layercake.net.process_update_from_packet(self.outgoing_packet_queue[0]["dest"],
                                                      self.outgoing_packet_queue[0][
                                                          "dest_position"], self.outgoing_packet_queue[0]["through"],
                                                      self.outgoing_packet_queue[0]["through_position"])
        ALOHA.transmit(self)

    def can_i_help(self, packet):
        """ A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        :param packet:
        """
        # Is this a packet already "directed" to me? I'm not checking if only
        # dest its me to avoid current protocol errors. Should be revised.
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

    def on_new_packet_received(self, incoming_packet):
        """ Function called from the lower layers when a packet is received.
        :param incoming_packet:
        """
        self.incoming_packet = incoming_packet

        if self.can_i_help(self.incoming_packet):
            if self.incoming_packet["through"][0:3] == "ANY":
                self.fsm.process("got_RTS")
            else:
                self.fsm.process(
                    self.packet_signal[self.incoming_packet["type"]])
        else:
            ALOHA.overhearing(self)

    def print_msg(self, msg):
        """

        :param msg:
        """
        pass
        # print "ALOHA4FBR (%s): %s at t=" % (self.layercake.hostname, msg),
        # Sim.now(), self.fsm.input_symbol, self.fsm.current_state


class DACAP(MAC):
    """DACAP : Distance Aware Collision Avoidance Protocol coupled with power control
    """

    def __init__(self, layercake, config):
        self.layercake = layercake
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)

        self.initialise_state_machine()
        self.timer = self.InternalTimer(self.fsm)
        self.TimerRequest = Sim.SimEvent("TimerRequest")

        self.outgoing_packet_queue = []
        self.incoming_packet = None
        self.last_packet = None

        self.packet_signal = {"ACK": "got_ACK", "RTS": "got_RTS",
                              "CTS": "got_CTS", "DATA": "got_DATA", "WAR": "got_WAR"}
        self.ack_packet_length = config["ack_packet_length"]
        self.rts_packet_length = config["rts_packet_length"]
        self.cts_packet_length = config["cts_packet_length"]
        self.war_packet_length = config["war_packet_length"]
        self.data_packet_length = config["data_packet_length"]

        self.max_transmission_attempts = config["attempts"]

        self.transmission_attempts = 0
        self.max_wait_to_retransmit = config["max2resend"]
        self.channel_access_retries = 0
        self.total_channel_access_attempts = 0
        self.next_timeout = 0
        self.pending_packet_ID = None

        # DACAP specific parameters
        self.T = 0  # It will be adapted according to the transmission range

        self.t_min = config["tminper"]
        self.Tw_min = config["twminper"]

        self.t_control = self.rts_packet_length / \
                         (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)

        self.deltaTData = config["deltatdata"]
        self.deltaD = config["deltadt"]

    def activate(self):
        """


        """
        Sim.activate(self.timer, self.timer.lifecycle(self.TimerRequest))

    class InternalTimer(Sim.Process):

        """

        :param fsm:
        """

        def __init__(self, fsm):
            Sim.Process.__init__(self, name="MAC_Timer")
            random.seed()
            self.fsm = fsm

        def lifecycle(self, request):
            """

            :param request:
            """
            while True:
                yield Sim.waitevent, self, request
                yield Sim.hold, self, request.signalparam[0]
                if self.interrupted():
                    self.interruptReset()
                else:
                    if not request.occurred:
                        self.fsm.process(request.signalparam[1])

    def initialise_state_machine(self):
        """initialise_state_machine:  set up Finite State Machine for RTSCTS
        """
        self.fsm = FSM("READY_WAIT", [], self)

        # Set default to Error
        self.fsm.set_default_transition(self.on_error, "READY_WAIT")

        # Transitions from READY_WAIT
        # Normal transitions
        self.fsm.add_transition(
            "send_DATA", "READY_WAIT", self.send_rts, "READY_WAIT")
        self.fsm.add_transition(
            "got_RTS", "READY_WAIT", self.check_rts, "READY_WAIT")
        self.fsm.add_transition(
            "got_X", "READY_WAIT", self.x_overheard, "BACKOFF")
        # Strange but possible transitions
        self.fsm.add_transition(
            "got_DATA", "READY_WAIT", self.check_pending_data, "READY_WAIT")
        self.fsm.add_transition(
            "got_ACK", "READY_WAIT", self.check_pending_ack, "READY_WAIT")
        self.fsm.add_transition(
            "got_WAR", "READY_WAIT", self.ignore_war, "READY_WAIT")
        self.fsm.add_transition(
            "got_CTS", "READY_WAIT", self.ignore_cts, "READY_WAIT")

        # Transitions from WAIT_CTS
        # Normal transitions
        self.fsm.add_transition(
            "send_DATA", "WAIT_CTS", self.queue_data, "WAIT_CTS")
        self.fsm.add_transition(
            "got_CTS", "WAIT_CTS", self.process_cts, "WAIT_TIME")
        self.fsm.add_transition(
            "got_RTS", "WAIT_CTS", self.ignore_rts, "WAIT_CTS")
        self.fsm.add_transition(
            "got_X", "WAIT_CTS", self.x_overheard, "WAIT_CTS")
        self.fsm.add_transition(
            "timeout", "WAIT_CTS", self.on_cts_timeout, "READY_WAIT")
        self.fsm.add_transition(
            "backoff", "WAIT_CTS", self.x_overheard, "BACKOFF")
        # It was a duplicated packet.
        self.fsm.add_transition(
            "got_ACK", "WAIT_CTS", self.on_transmit_success, "READY_WAIT")
        # Strange but possible transitions:
        # The CTS had collided, that's why I'm still here.
        self.fsm.add_transition(
            "got_WAR", "WAIT_CTS", self.x_overheard, "BACKOFF")

        # Transitions from WAIT_TIME
        self.fsm.add_transition(
            "send_DATA", "WAIT_TIME", self.queue_data, "WAIT_TIME")
        self.fsm.add_transition(
            "got_RTS", "WAIT_TIME", self.ignore_rts, "WAIT_TIME")
        self.fsm.add_transition(
            "got_WAR", "WAIT_TIME", self.process_war, "WAIT_TIME")
        self.fsm.add_transition(
            "got_ACK", "WAIT_TIME", self.ignore_ack, "WAIT_TIME")

        self.fsm.add_transition(
            "got_X", "WAIT_TIME", self.x_overheard, "WAIT_TIME")
        self.fsm.add_transition(
            "timeout", "WAIT_TIME", self.transmit, "WAIT_ACK")
        self.fsm.add_transition(
            "backoff", "WAIT_TIME", self.x_overheard, "BACKOFF")

        # Transitions from WAIT_DATA: I can receive RTS from other nodes, the
        # DATA packet that I expect or overhear other communications
        self.fsm.add_transition(
            "send_DATA", "WAIT_DATA", self.queue_data, "WAIT_DATA")
        self.fsm.add_transition(
            "got_RTS", "WAIT_DATA", self.x_overheard, "WAIT_DATA")
        self.fsm.add_transition(
            "got_DATA", "WAIT_DATA", self.on_data_reception, "READY_WAIT")

        self.fsm.add_transition(
            "got_X", "WAIT_DATA", self.x_overheard, "WAIT_DATA")
        self.fsm.add_transition(
            "timeout", "WAIT_DATA", self.on_data_timeout, "READY_WAIT")
        self.fsm.add_transition(
            "backoff", "WAIT_DATA", self.x_overheard, "BACKOFF")

        # Transitions from WAIT_ACK: I can receive RTS from other nodes, the
        # ACK packet that I expect or overhear other communications
        self.fsm.add_transition(
            "send_DATA", "WAIT_ACK", self.queue_data, "WAIT_ACK")
        self.fsm.add_transition(
            "got_RTS", "WAIT_ACK", self.ignore_rts, "WAIT_ACK")
        self.fsm.add_transition(
            "got_ACK", "WAIT_ACK", self.on_transmit_success, "READY_WAIT")
        # This should not happen
        self.fsm.add_transition(
            "got_WAR", "WAIT_ACK", self.process_war, "WAIT_ACK")

        self.fsm.add_transition(
            "got_X", "WAIT_ACK", self.x_overheard, "BACKOFF")
        self.fsm.add_transition(
            "timeout", "WAIT_ACK", self.on_ack_timeout, "READY_WAIT")

        # Transitions from BACKOFF
        self.fsm.add_transition(
            "send_DATA", "BACKOFF", self.queue_data, "BACKOFF")
        self.fsm.add_transition(
            "got_RTS", "BACKOFF", self.process_rts, "BACKOFF")
        self.fsm.add_transition(
            "got_CTS", "BACKOFF", self.ignore_cts, "BACKOFF")
        self.fsm.add_transition(
            "got_DATA", "BACKOFF", self.on_data_reception, "BACKOFF")
        self.fsm.add_transition(
            "got_ACK", "BACKOFF", self.on_transmit_success, "READY_WAIT")
        self.fsm.add_transition(
            "got_WAR", "BACKOFF", self.process_war, "BACKOFF")

        self.fsm.add_transition("got_X", "BACKOFF", self.x_overheard, "BACKOFF")
        self.fsm.add_transition(
            "timeout", "BACKOFF", self.on_timeout, "READY_WAIT")
        self.fsm.add_transition("accept", "BACKOFF", self.send_cts, "WAIT_DATA")
        self.fsm.add_transition(
            "backoff", "BACKOFF", self.x_overheard, "BACKOFF")

    def x_overheard(self):
        """ By overhearing the channel, we can obtain valuable information.
        """
        if self.fsm.current_state == "READY_WAIT":
            if self.incoming_packet["type"] != "RTS" and self.incoming_packet["type"] != "CTS":
                self.logger.debug(
                    "I'm in READY_WAIT and have received " + self.incoming_packet["type"])

        elif self.fsm.current_state == "WAIT_CTS":
            if self.incoming_packet["type"] == "CTS":
                k = self.T * self.t_min
            elif self.incoming_packet["type"] == "RTS":
                k = self.T
            else:
                # Let's think: Can I overhear something else? I think I
                # shouldn't
                self.logger.debug(
                    "I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
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
                # Let's think: Can I overhear something else? I think I
                # shouldn't
                self.logger.debug(
                    "I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
                return True

            if (Sim.now() - self.time) < k:
                self.fsm.process("backoff")
                return True
            else:
                return False

        elif self.fsm.current_state == "WAIT_DATA":
            if self.incoming_packet["type"] == "RTS" and self.incoming_packet["ID"] == self.pending_packet_ID:
                # It is the same RTS or the same with a higher power value - It
                # doesn't make sense but may happen
                return True
            else:
                if self.incoming_packet["type"] == "RTS":
                    k = 2 * self.T - self.T * self.t_min
                elif self.incoming_packet["type"] == "CTS":
                    k = 2 * self.T - self.T * self.Tw_min
                else:
                    # Let's think: Can I overhear something else? I think I
                    # shouldn't
                    self.logger.debug(
                        "I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
                    return True

                if (Sim.now() - self.time) < k:
                    self.send_warning()
                    self.fsm.process("backoff")
                    return True
                else:
                    return False

        self.last_packet = self.incoming_packet

        self.t_data = self.incoming_packet[
                          'length'] / (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
        t = self.layercake.phy.level2delay(self.incoming_packet['level'])

        # If I am here, that means that I already was in the backoff state. I
        # have just to schedule the backoff timer
        if self.incoming_packet["type"] == "WAR":
            backoff = self.get_backoff_time(
                self.incoming_packet["type"], t)  # This is new!
        elif self.incoming_packet["type"] == "CTS" or self.incoming_packet["type"] == "SIL":
            backoff = self.get_backoff_time(self.incoming_packet["type"], t)
        elif self.incoming_packet["type"] == "RTS" or self.incoming_packet["type"] == "DATA":
            backoff = self.get_backoff_time(self.incoming_packet["type"], t)
        elif self.incoming_packet["type"] == "ACK":
            backoff = 0.0  # I'm done

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        if DEBUG:
            self.logger.debug("Sleep for " + str(backoff) + " due to " + self.incoming_packet["type"]
                              + " coming from " +
                              self.incoming_packet["source"]
                              + " to " + self.incoming_packet["dest"]
                              + " through " + self.incoming_packet["through"])

        self.TimerRequest.signal((backoff, "timeout"))
        self.next_timeout = Sim.now() + backoff

        self.incoming_packet = None

    def ignore_rts(self):
        """


        """
        self.logger.debug(
            "I can't attend the RTS received from " + self.incoming_packet["source"])
        self.incoming_packet = None

    def ignore_cts(self):
        """


        """
        self.logger.debug(
            "Ignoring CTS coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def ignore_ack(self):
        """


        """
        self.logger.debug(
            "Ignoring ACK coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def ignore_war(self):
        """


        """
        self.logger.debug(
            "Ignoring WAR coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def check_pending_data(self):
        """


        """
        if self.pending_packet_ID is not None:
            if self.incoming_packet["ID"] == self.pending_packet_ID:
                self.logger.warn(
                    "Despite everything, I properly received the DATA packet from: " + self.incoming_packet["source"])
                self.on_data_reception()
        else:
            self.on_error()

    def check_pending_ack(self):
        """


        """
        if self.pending_packet_ID is not None:
            if self.incoming_packet["ID"] == self.pending_packet_ID:
                self.logger.warn(
                    "Despite everything, I we properly transmitted to: " + self.incoming_packet["source"])
                self.on_transmit_success()
        else:
            self.on_error()

    def send_rts(self):
        """ Request To Send.
        """
        self.level = self.outgoing_packet_queue[0]["level"]
        self.T = self.layercake.phy.level2delay(self.level)

        if self.layercake.phy.is_idle():
            self.total_channel_access_attempts += 1
            self.transmission_attempts += 1
            self.channel_access_retries = 0

            if self.outgoing_packet_queue[0]["through"].startswith("ANY"):
                self.multicast = True
            else:
                self.multicast = False

            rts_packet = {"type": "RTS", "ID": self.outgoing_packet_queue[0]["ID"],
                          "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                          "dest": self.outgoing_packet_queue[0]["dest"],
                          "dest_position": self.outgoing_packet_queue[0]["dest_position"],
                          "through": self.outgoing_packet_queue[0]["through"],
                          "through_position": self.outgoing_packet_queue[0]["through_position"],
                          "length": self.rts_packet_length, "level": self.outgoing_packet_queue[0]["level"],
                          "time_stamp": Sim.now()}

            if DEBUG:
                self.logger.debug("Transmitting RTS to " + self.outgoing_packet_queue[0]["dest"] + " through " +
                                  self.outgoing_packet_queue[
                                      0]["through"] + " with power level " + str(
                    self.outgoing_packet_queue[0]["level"]))

            self.level = self.outgoing_packet_queue[0]["level"]
            self.T = self.layercake.phy.level2delay(self.level)

            self.layercake.phy.transmit_packet(rts_packet)

            timeout = self.get_timeout("CTS", self.T)
            self.TimerRequest.signal((timeout, "timeout"))
            self.time = Sim.now()
            self.pending_packet_ID = self.outgoing_packet_queue[0]["ID"]
            self.fsm.current_state = "WAIT_CTS"
        else:
            self.channel_access_retries += 1
            timeout = random.random() * (2 * self.T + self.t_data)
            self.TimerRequest.signal((timeout, self.fsm.input_symbol))

    def process_rts(self):
        """ Maybe I can do it now.
        """
        if self.last_packet["type"] == "CTS" and self.last_packet["through"] == self.incoming_packet["source"]:
            # I was sleeping because one of my neighbors was receiving a packet
            # and now it wants to forward it.
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")
        elif self.last_packet["type"] == "DATA" and self.last_packet["through"] == self.incoming_packet["source"] and \
                        self.last_packet["dest"] == self.incoming_packet["dest"]:
            # I was sleeping because one of my neighbors was receiving a packet
            # which I also overheard.
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")

    def get_wait_time(self, t, u):
        """ Returns the time to wait before transmitting
        :param t:
        :param u:
        """
        t1 = (self.t_min * t -
              min(self.deltaD * t, self.t_data, 2 * t - self.t_min * t)) / 2.0
        t2 = (self.t_min * t - self.deltaTData) / 2.0
        t3 = (self.t_min * t + self.Tw_min * t - 2 * self.deltaD * t) / 4.0

        if t1 <= u <= t2:
            timewait = 2 * (u + self.deltaD * t) - self.t_min * t
            self.logger.debug(
                "First case, u=" + str(u) + ", I'll wait for " + str(timewait))
        elif u > max(t2, min(t1, t3)):
            timewait = 2 * (u + self.deltaD * t) - self.Tw_min * t
            self.logger.debug(
                "Second case, u=" + str(u) + ", I'll wait for " + str(timewait))
        else:
            timewait = self.t_min * t - 2 * u
            self.logger.debug(
                "Third case, u=" + str(u) + ", I'll wait for " + str(timewait))

        if timewait <= max(2 * self.deltaD * t, self.Tw_min * t):
            self.logger.debug(
                "Fourth case, u=" + str(u) + ", I'll wait for " + str(max(2 * self.deltaD * t, self.Tw_min * t)))
            return max(2 * self.deltaD * t, self.Tw_min * t)
        else:
            return timewait

    def get_backoff_time(self, packet_type, t):
        """ Returns the backoff for a specific state.
        :param packet_type:
        :param t:
        """
        if packet_type == "RTS" or packet_type == "WAR":
            backoff = 2 * t + 2 * t - self.Tw_min * t
        elif packet_type == "CTS" or packet_type == "SIL":
            backoff = 2 * t + 2 * t - self.Tw_min * t + self.t_data
        elif packet_type == "DATA":
            backoff = 2 * t

        self.logger.warning("Backoff for {backoff} for {type} packet based on t{t},Tw{TW},Td{TD}".format(
            backoff=backoff,
            type=packet_type,
            T=t,
            TW=self.Tw_min,
            TD=self.t_data))
        return backoff

    def get_timeout(self, packet_type, t):
        """ Returns the timeout for a specific state.
        :param packet_type:
        :param t:
        """
        if packet_type == "CTS":
            return 2 * t + 2 * self.t_control
        elif packet_type == "DATA":
            return 2 * t + 2 * t - self.Tw_min * t + 2 * self.t_data + 2 * self.t_control
        elif packet_type == "ACK":
            return 2 * t + self.t_data + self.t_control

    def process_cts(self):
        """ The CTS has been received.
        """
        self.logger.debug("CTS from " + self.incoming_packet[
            "source"] + " coming from " + self.incoming_packet["through"] + " properly received.")

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        u = (Sim.now() - self.incoming_packet["time_stamp"]) / 2.0
        timewait = self.get_wait_time(self.T, u)

        self.logger.debug("Waiting for " + str(timewait) + " due to " + self.incoming_packet["type"]
                          + " coming from " + self.incoming_packet["source"]
                          + " through " + self.incoming_packet["through"])

        self.TimerRequest.signal((timewait, "timeout"))

        self.incoming_packet = None

    def send_warning(self):
        """ If the next best hop has been already selected but a CTS has been received, we should indicate to the sender that
            it is not necessary anymore. Otherwise, implicit WAR can be used.
        """
        war_packet = {"type": "WAR",
                      "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                      "dest": self.last_cts["dest"], "dest_position": self.last_cts["dest_position"],
                      "through": self.last_cts["dest"], "through_position": self.last_cts["dest_position"],
                      "length": self.war_packet_length, "level": self.last_cts["level"]}

        self.logger.debug(
            "Transmitting Warning Packet to " + self.last_cts["dest"])

        self.layercake.phy.transmit_packet(war_packet)

    def process_war(self):
        """ I should defer the packets for the same receiver.
        """
        if self.fsm.current_state == "WAIT_ACK":
            self.logger.debug(
                "Well, in any case, I wait for the ACK as I have just transmitted.")
            self.incoming_packet = None
            return True
        elif self.fsm.current_state != "READY_WAIT" and self.fsm.current_state is not "WAIT_ACK":
            self.logger.debug(
                "Ok, then maybe later on we will talk about this " + self.incoming_packet["source"])
            self.fsm.process("backoff")

    def check_rts(self):
        """ Before proceeding with the data transmission, just check if it's a duplicated packet (maybe the ACK collided).
        """
        if self.layercake.net.have_duplicate_packet(self.incoming_packet):
            packet_origin = [
                self.incoming_packet["source"], self.incoming_packet["source_position"]]
            self.send_ack(packet_origin)
        else:
            self.send_cts()
            self.fsm.current_state = "WAIT_DATA"

    def send_cts(self):
        """ Clear To Send: I'm proposing myself as a good candidate for the next transmission or I just let transmit if I have been
        already selected.
        """
        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        cts_packet = {"type": "CTS",
                      "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                      "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                      "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                      "length": self.cts_packet_length, "rx_energy": self.layercake.phy.rx_energy,
                      "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"],
                      "ID": self.incoming_packet["ID"]}

        if DEBUG:
            self.logger.debug(
                "Transmitting CTS to " + self.incoming_packet["source"])
        self.pending_packet_ID = self.incoming_packet["ID"]
        # I may need this if I have to send a warning packet
        self.last_cts = cts_packet
        self.layercake.phy.transmit_packet(cts_packet)

        self.level = self.incoming_packet["level"]
        self.T = self.layercake.phy.level2delay(self.level)

        timeout = self.get_timeout("DATA", self.T)
        self.TimerRequest.signal((timeout, "timeout"))
        self.time = Sim.now()

        self.incoming_packet = None

    def send_ack(self, packet_origin):
        """ Sometimes we can not use implicit ACKs.
        :param packet_origin:
        """
        if DEBUG:
            self.logger.debug("ACK to {}".format(packet_origin[0]))

        ack_packet = {"type": "ACK",
                      "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                      "dest": packet_origin[0], "dest_position": packet_origin[1],
                      "through": packet_origin[0], "through_position": packet_origin[1],
                      "length": self.ack_packet_length, "level": self.incoming_packet["level"],
                      "ID": self.incoming_packet["ID"]}

        self.layercake.phy.transmit_packet(ack_packet)

    def initiate_transmission(self, outgoing_packet):
        """ Function called from the upper layers to transmit a packet.
        :param outgoing_packet:
        """
        self.outgoing_packet_queue.append(outgoing_packet)
        self.fsm.process("send_DATA")

    def on_new_packet_received(self, incoming_packet):
        """ Function called from the lower layers when a packet is received.
        :param incoming_packet:
        """
        self.incoming_packet = incoming_packet
        if self.can_i_help(incoming_packet):
            self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            self.overhearing()

    def can_i_help(self, packet):
        """ A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        :param packet:
        """
        # Is this a packet already "directed" to me? I'm not checking if only
        # dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTSs are the only types of packet that are directly addressed
        if packet["type"] == "CTS" and packet["dest"] == self.layercake.hostname:
            return True

        return False

    def overhearing(self):
        """ Valuable information can be obtained from overhearing the channel.
        """
        if self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet[
                "dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")
        else:
            self.fsm.process("got_X")

    def on_data_reception(self):
        """ After the RTS/CTS exchange, a data packet is received. We should acknowledge the previous node or we can try to use implicit
        acknowledges if the that does not mean a waste of power.
        """
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.fsm.current_state != "BACKOFF":
            self.last_packet = None

        # We should acknowledge the previous node, this is not an END to END
        # ACK
        packet_origin = self.incoming_packet["route"][-1]

        if self.layercake.net.need_explicit_ack(self.incoming_packet["level"], self.incoming_packet["dest"]) or len(
                self.outgoing_packet_queue) != 0:
            self.send_ack(packet_origin)

        self.layercake.net.on_packet_reception(self.incoming_packet)

        self.pending_packet_ID = None
        self.incoming_packet = None

    def on_error(self):
        """ An unexpected transition has been followed. This should not happen.
        """
        self.logger.debug("ERROR!")
        print self.layercake.hostname, self.incoming_packet, self.fsm.input_symbol, self.fsm.current_state

    def on_transmit_success(self):
        """ When an ACK is received, we can assume that everything has gone fine, so it's all done.
        """
        if DEBUG:
            self.logger.debug(
                "Successfully Transmitted to " + self.incoming_packet["source"])
        self.pending_packet_ID = None

        # We got an ACK, we should stop the timer.
        p = Sim.Process()
        p.interrupt(self.timer)

        self.post_success_or_fail()

    def on_transmit_fail(self):
        """ All the transmission attemps have been completed. It's impossible to reach the node.
        """
        self.logger.debug(
            "Failed to transmit to " + self.outgoing_packet_queue[0]["dest"])
        self.post_success_or_fail()

    def post_success_or_fail(self):
        """ Successfully or not, we have finished the current transmission.
        """
        self.outgoing_packet_queue.pop(0)["dest"]
        self.transmission_attempts = 0

        # Is there anything else to do?
        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0

    def on_ack_timeout(self):
        """ The timeout has experied and NO ACK has been received.
        """
        self.transmission_attempts += 1
        self.logger.debug("Timed Out, No Ack Received")

        if self.transmission_attempts < self.max_transmission_attempts:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))

    def on_data_timeout(self):
        """ The timeout has experied and NO DATA has been received.
        """
        self.logger.debug("Timed Out!, No Data Received")

        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0

    def on_cts_timeout(self):
        """


        """
        self.transmission_attempts += 1
        self.logger.debug("Timed Out, No CTS Received")
        if self.layercake.phy.collision_detected():
            self.logger.debug("It seems that there has been a collision.")

        if self.transmission_attempts > self.max_transmission_attempts:
            self.logger.debug("Sorry, I cannot do anything else.")
        else:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))

    def on_timeout(self):
        """


        """
        self.logger.debug("Exiting from back off")

        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0

    def transmit(self):
        """


        """
        self.logger.debug("transmit to " + self.outgoing_packet_queue[0][
            "dest"] + " through " + self.outgoing_packet_queue[0]["through"])
        self.layercake.phy.transmit_packet(self.outgoing_packet_queue[0])

        timeout = self.get_timeout("ACK", self.T)
        self.TimerRequest.signal((timeout, "timeout"))

    def queue_data(self):
        """


        """
        self.logger.debug("Queuing Data")


class DACAP4FBR(DACAP):
    """DACAP for FBR: adaptation of DACAP with power control to couple it with FBR
    """

    def __init__(self, node, config):

        DACAP.__init__(self, node, config)

        # A new packet type
        self.packet_signal["SIL"] = "got_SIL"
        self.SIL_packet_length = config["SIL_packet_length"]

        self.valid_candidates = {}
        self.original_through = None

        # By MC_RTS we refer to MultiCast RTS packets. We should consider to
        # process them.
        self.fsm.add_transition(
            "got_MC_RTS", "READY_WAIT", self.process_mc_rts, "AWARE")
        self.fsm.add_transition(
            "got_SIL", "READY_WAIT", self.ignore_sil, "READY_WAIT")
        self.fsm.add_transition(
            "defer", "READY_WAIT", self.ignore_war, "READY_WAIT")

        # Definition of a new state, AWARE, and its transitions
        self.fsm.add_transition("send_CTS", "AWARE", self.send_cts, "WAIT_DATA")
        self.fsm.add_transition(
            "ignore_RTS", "AWARE", self.x_overheard, "BACKOFF")

        # Now I am receiving several CTS, then, I should just append them
        # Maybe I should tell him to be silent
        self.fsm.add_transition(
            "got_MC_RTS", "WAIT_CTS", self.process_mc_rts, "WAIT_CTS")
        self.fsm.add_transition(
            "got_CTS", "WAIT_CTS", self.append_cts, "WAIT_CTS")
        self.fsm.add_transition(
            "got_WAR", "WAIT_CTS", self.process_war, "WAIT_CTS")
        self.fsm.add_transition(
            "got_SIL", "WAIT_CTS", self.process_sil, "BACKOFF")

        self.fsm.add_transition(
            "timeout", "WAIT_CTS", self.select_cts, "WAIT_CTS")
        self.fsm.add_transition(
            "transmit", "WAIT_CTS", self.process_cts, "WAIT_TIME")
        self.fsm.add_transition(
            "retransmit", "WAIT_CTS", self.send_rts, "WAIT_CTS")
        self.fsm.add_transition(
            "abort", "WAIT_CTS", self.on_transmit_fail, "READY_WAIT")
        self.fsm.add_transition(
            "defer", "WAIT_CTS", self.x_overheard, "BACKOFF")

        # It may be the case that I still receive CTS when being in WAIT_TIME
        self.fsm.add_transition(
            "got_WAR", "WAIT_TIME", self.process_war, "WAIT_TIME")
        self.fsm.add_transition(
            "got_SIL", "WAIT_TIME", self.process_sil, "BACKOFF")
        self.fsm.add_transition(
            "got_MC_RTS", "WAIT_TIME", self.process_mc_rts, "WAIT_TIME")
        self.fsm.add_transition(
            "got_CTS", "WAIT_TIME", self.ignore_cts, "WAIT_TIME")
        self.fsm.add_transition(
            "timeout", "WAIT_TIME", self.select_cts, "WAIT_TIME")
        self.fsm.add_transition(
            "transmit", "WAIT_TIME", self.transmit, "WAIT_ACK")
        self.fsm.add_transition(
            "retransmit", "WAIT_TIME", self.send_rts, "WAIT_CTS")
        self.fsm.add_transition(
            "abort", "WAIT_TIME", self.on_transmit_fail, "READY_WAIT")
        self.fsm.add_transition(
            "defer", "WAIT_TIME", self.x_overheard, "BACKOFF")

        # New transition from WAIT_DATA: I have been not selected as the next
        # hop
        self.fsm.add_transition(
            "got_MC_RTS", "WAIT_DATA", self.process_mc_rts, "WAIT_DATA")
        self.fsm.add_transition(
            "ignored", "WAIT_DATA", self.x_overheard, "BACKOFF")
        self.fsm.add_transition(
            "got_CTS", "WAIT_DATA", self.ignore_cts, "WAIT_DATA")
        # From maybe previous transmissions - check it
        self.fsm.add_transition(
            "got_WAR", "WAIT_DATA", self.ignore_war, "WAIT_DATA")
        self.fsm.add_transition(
            "got_SIL", "WAIT_DATA", self.ignore_sil, "WAIT_DATA")

        # From maybe previous transmissions - check it
        self.fsm.add_transition(
            "got_WAR", "BACKOFF", self.ignore_war, "BACKOFF")
        self.fsm.add_transition(
            "got_MC_RTS", "BACKOFF", self.process_mc_rts, "BACKOFF")
        self.fsm.add_transition(
            "got_SIL", "BACKOFF", self.ignore_sil, "BACKOFF")
        self.fsm.add_transition("defer", "BACKOFF", self.x_overheard, "BACKOFF")

        self.fsm.add_transition(
            "got_MC_RTS", "WAIT_ACK", self.process_mc_rts, "WAIT_ACK")

    def ignore_sil(self):
        """


        """
        self.logger.debug(
            "Ignoring SIL coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def process_mc_rts(self):
        """ Someone is looking for help, may I help? Now the active nodes are the ones that transmit, maybe we should do it the opposite way
        """
        if self.fsm.current_state == "AWARE":
            if self.layercake.net.i_am_a_valid_candidate(self.incoming_packet):
                if self.layercake.net.have_duplicate_packet(self.incoming_packet):
                    packet_origin = [
                        self.incoming_packet["source"], self.incoming_packet["source_position"]]
                    self.send_ack(packet_origin)
                    self.fsm.current_state = "READY_WAIT"
                else:
                    self.fsm.process("send_CTS")
            else:
                self.logger.debug("I can't attend the MultiCast RTS received from " +
                                  self.incoming_packet["source"] + " but I will be silent.")
                self.fsm.process("ignore_RTS")
                # The SIlence part is being revised.
                # elif self.fsm.current_state == "WAIT_DATA":
                # if self.last_cts["dest"] == self.incoming_packet["source"]:
                # self.ignore_rts()
                # else:
                # self.send_silence()
                # elif self.fsm.current_state == "WAIT_CTS" or self.fsm.current_state == "WAIT_TIME":
                # self.send_silence()
                # elif self.fsm.current_state == "BACKOFF":

    def send_silence(self):
        """ Please be quiet!
        """
        sil_packet = {"type": "SIL",
                      "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                      "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                      "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                      "length": self.SIL_packet_length, "tx_energy": self.layercake.phy.tx_energy,
                      "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

        self.logger.debug(
            "Transmitting SIL to " + self.incoming_packet["source"])

        self.layercake.phy.transmit_packet(sil_packet)

        self.incoming_packet = None

    def process_sil(self):
        """ The next time that I try to find a route I should do it starting again from ANY0.
        """
        self.outgoing_packet_queue[0]["through"] = "ANY0"
        self.outgoing_packet_queue[0]["level"] = 0
        self.x_overheard()

    def append_cts(self):
        """ More than one CTS is received when looking for the next best hop. We should consider all of them.
        The routing layer decides.
        """
        self.valid_candidates[self.incoming_packet["through"]] = (
                                                                     Sim.now() - self.incoming_packet[
                                                                         "time_stamp"]) / 2.0, self.incoming_packet[
                                                                     "rx_energy"], self.incoming_packet[
                                                                     "through_position"]
        self.logger.debug("Appending CTS to " + self.incoming_packet[
            "source"] + " coming from " + self.incoming_packet["through"])
        self.layercake.net.add_node(
            self.incoming_packet["through"], self.incoming_packet["through_position"])

        self.incoming_packet = None

        if not self.multicast:
            # Update the timer: 1.-Stop, 2.-Restart
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("timeout")

    def select_cts(self):
        """ Once we have wait enough, that is, 2 times the distance at which the best next hop should be, we should select it.
        """
        current_through = self.outgoing_packet_queue[0]["through"]
        self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0][
            "through_position"] = self.layercake.net.select_route(
            self.valid_candidates, self.outgoing_packet_queue[0]["through"], self.transmission_attempts,
            self.outgoing_packet_queue[0]["dest"])

        if self.outgoing_packet_queue[0]["through"] == "ABORT":
            # We have consumed all the attemps
            self.fsm.process("abort")

        elif self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
            # We should retransmit increasing the power
            self.outgoing_packet_queue[0]["level"] = int(
                self.outgoing_packet_queue[0]["through"][3])
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "2CHANCE":
            # We should give a second chance to current node
            self.outgoing_packet_queue[0]["through"] = current_through
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"][0:5] == "NEIGH":
            # With current transmission power level, the destination has become
            # a neighbor
            self.outgoing_packet_queue[0]["level"] = int(
                self.outgoing_packet_queue[0]["through"][5])
            self.outgoing_packet_queue[0][
                "through"] = self.outgoing_packet_queue[0]["dest"]
            self.outgoing_packet_queue[0][
                "through_position"] = self.outgoing_packet_queue[0]["dest_position"]
            self.fsm.process("retransmit")

        else:
            self.fsm.process("transmit")

    def process_cts(self):
        """ Once a candidate has been selected, we should wait before transmitting according to DACAP.
        """
        u = self.valid_candidates[self.outgoing_packet_queue[0]["through"]][0]

        if self.multicast:
            # Or zero...
            timewait = self.get_wait_time(self.T, u) - (2 * self.T - u)
        else:
            timewait = self.get_wait_time(self.T, u)

        self.logger.debug("Waiting for " + str(timewait) +
                          " before Transmitting through " + self.outgoing_packet_queue[0]["through"])
        self.TimerRequest.signal((timewait, "timeout"))

    def transmit(self):
        """ Real Transmission of the Packet.
        """
        self.layercake.net.process_update_from_packet(self.outgoing_packet_queue[0]["dest"],
                                                      self.outgoing_packet_queue[0][
                                                          "dest_position"], self.outgoing_packet_queue[0]["through"],
                                                      self.outgoing_packet_queue[0]["through_position"])
        DACAP.transmit(self)

    def overhearing(self):
        """ Valuable information can be obtained from overhearing the channel.
        """
        if self.incoming_packet["type"] == "DATA" and self.fsm.current_state == "WAIT_DATA":
            if self.pending_packet_ID == self.incoming_packet["ID"]:
                # I have not been selected as the next hop. I read it from the
                # data.
                self.fsm.process("ignored")

        elif self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet[
                "dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")

        elif self.incoming_packet["type"] == "CTS" and self.fsm.current_state == "WAIT_DATA" and self.last_cts[
            "dest"] == self.incoming_packet["dest"]:
            # This is another candidate proposing himself as a good candidate
            self.fsm.process("got_CTS")

        elif self.incoming_packet["type"] == "SIL":
            return False

        else:
            self.fsm.process("got_X")

    def can_i_help(self, packet):
        """ A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        :param packet:
        """
        # Is this a packet already "directed" to me? I'm not checking if only
        # dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTS, ACK, WAR and SIL are the only types of packet that are directly
        # addressed
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

    def on_new_packet_received(self, incoming_packet):
        """ Function called from the lower layers when a packet is received.
        :param incoming_packet:
        """
        self.incoming_packet = incoming_packet

        if self.can_i_help(self.incoming_packet):
            if self.incoming_packet["through"][0:3] == "ANY":
                self.fsm.process("got_MC_RTS")
            else:
                self.fsm.process(
                    self.packet_signal[self.incoming_packet["type"]])
        else:
            self.overhearing()

    def ignore_war(self):
        """ If I'm already in back off due to a SIL packet, I just ignore the WAR packets that I receive.
        """
        self.logger.debug(
            "Ok, Ok, I already knew that " + self.incoming_packet["source"])
        self.incoming_packet = None

    def process_war(self):
        """ Taking into account that more than one reply can be received, it may be the case
        just some of them send a warning packet but not all, so, only those are discarded.
        """
        self.logger.debug(
            "Ok, I won't consider you " + self.incoming_packet["source"])

        try:
            del self.valid_candidates[self.incoming_packet["source"]]
        except KeyError:
            self.logger.debug(
                "You were not in my list " + self.incoming_packet["source"] + ". Your packet may have collided.")

        if len(self.valid_candidates) == 0:
            if self.multicast == True and self.fsm.current_state == "WAIT_TIME":
                # All candidates have sent a WAR packet or the only one did
                self.fsm.process("defer")
            elif self.multicast == False and self.fsm.current_state == "WAIT_TIME":
                self.fsm.process("defer")

        self.incoming_packet = None

    def send_rts(self):
        """ The RTS sent is the normal one, but we should initialize the list of replies.
        """
        self.valid_candidates = {}
        DACAP.send_rts(self)


class CSMA(MAC):
    """CSMA: Carrier Sensing Multiple Access - Something between ALOHA and DACAP

    This implementation is extremely close to MACA (Karn 1990)
    """

    def __init__(self, layercake, config):
        self.layercake = layercake
        self.config = config
        self.logger = getattr(self.layercake, "logger", None)
        if self.logger is not None:
            self.logger = self.logger.getChild("%s" % self.__class__.__name__)
        else:
            logging.basicConfig()
            self.logger = logging.getLogger("%s" % self.__class__.__name__)

        self.initialise_state_machine()
        self.timer = self.InternalTimer(self.fsm)
        self.TimerRequest = Sim.SimEvent("TimerRequest")

        self.outgoing_packet_queue = []
        self.incoming_packet = None
        self.last_packet = None

        self.packet_signal = {
            "ACK": "got_ACK", "RTS": "got_RTS", "CTS": "got_CTS", "DATA": "got_DATA"}

        self.transmission_attempts = 0
        self.ack_failures = 0
        self.channel_access_retries = 0
        self.total_channel_access_attempts = 0
        self.next_timeout = 0
        self.pending_data_packet_from = None
        self.pending_ack_packet_from = None
        self.multicast = False

        # Configs set at activate
        self.ack_packet_length = None
        self.rts_packet_length = None
        self.cts_packet_length = None
        self.data_packet_length = None
        self.max_wait_to_retransmit = None
        self.max_transmission_attempts = None
        # Timing parameters
        self.T = None
        self.t_data = None
        self.t_control = None


    def activate(self):
        """


        """
        self.ack_packet_length = self.config["ack_packet_length"]
        self.rts_packet_length = self.config["rts_packet_length"]
        self.cts_packet_length = self.config["cts_packet_length"]
        self.data_packet_length = self.config["data_packet_length"]
        self.max_wait_to_retransmit = self.config["max2resend"]
        self.max_transmission_attempts = self.config["attempts"]

        # Timing parameters
        self.T = self.layercake.phy.level2delay(0)  # It will be adapted according to the transmission range
        self.t_data = self.data_packet_length / \
                      (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
        self.t_control = self.rts_packet_length / \
                         (self.layercake.phy.bandwidth * 1e3 * self.layercake.phy.band2bit)
        Sim.activate(self.timer, self.timer.lifecycle(self.TimerRequest))

    class InternalTimer(Sim.Process):
        """

        :param fsm:
        """

        def __init__(self, fsm):
            Sim.Process.__init__(self, name="MAC_Timer")
            random.seed()
            self.fsm = fsm

        def lifecycle(self, request):
            """

            :param request:
            """
            while True:
                yield Sim.waitevent, self, request
                yield Sim.hold, self, request.signalparam[0]
                if self.interrupted():
                    self.interruptReset()
                else:
                    if not request.occurred:
                        self.fsm.process(request.signalparam[1])

    def initialise_state_machine(self):
        """initialise_state_machine: set up Finite State Machine for RTSCTS
        """
        self.fsm = FSM("READY_WAIT", [], self)

        # Set default to Error
        self.fsm.set_default_transition(self.on_error, "READY_WAIT")

        # Transitions from READY_WAIT
        # Normal transitions
        # Only if the channel is idle will I transmit
        self.fsm.add_transition(
            "send_DATA", "READY_WAIT", self.send_rts, "READY_WAIT")
        # Check if it is duplicated
        self.fsm.add_transition(
            "got_RTS", "READY_WAIT", self.check_rts, "READY_WAIT")
        self.fsm.add_transition("timeout", "READY_WAIT", self.on_timeout, "READY_WAIT")

        self.fsm.add_transition(
            "got_X", "READY_WAIT", self.x_overheard, "BACKOFF")
        # Strange but possible transitions
        self.fsm.add_transition(
            "got_DATA", "READY_WAIT", self.check_pending_data, "READY_WAIT")
        self.fsm.add_transition(
            "got_ACK", "READY_WAIT", self.check_pending_ack, "READY_WAIT")

        # Transitions from WAIT_CTS
        # Normal transitions
        self.fsm.add_transition(
            "send_DATA", "WAIT_CTS", self.queue_data, "WAIT_CTS")
        # This is the main difference with DACAP
        self.fsm.add_transition(
            "got_CTS", "WAIT_CTS", self.transmit, "WAIT_ACK")
        self.fsm.add_transition(
            "got_RTS", "WAIT_CTS", self.ignore_rts, "WAIT_CTS")
        self.fsm.add_transition(
            "timeout", "WAIT_CTS", self.on_cts_timeout, "READY_WAIT")

        self.fsm.add_transition(
            "got_X", "WAIT_CTS", self.x_overheard, "WAIT_CTS")
        self.fsm.add_transition(
            "backoff", "WAIT_CTS", self.x_overheard, "BACKOFF")
        # Strange but possible transitions
        # I was transmitting a duplicated packet
        self.fsm.add_transition(
            "got_ACK", "WAIT_CTS", self.on_transmit_success, "READY_WAIT")

        # Transitions from WAIT_DATA
        self.fsm.add_transition(
            "send_DATA", "WAIT_DATA", self.queue_data, "WAIT_DATA")
        self.fsm.add_transition(
            "got_RTS", "WAIT_DATA", self.ignore_rts, "WAIT_DATA")
        self.fsm.add_transition(
            "got_CTS", "WAIT_DATA", self.ignore_cts, "WAIT_DATA")
        self.fsm.add_transition(
            "got_DATA", "WAIT_DATA", self.on_data_reception, "WAIT_DATA")
        self.fsm.add_transition(
            "timeout", "WAIT_DATA", self.on_data_timeout, "READY_WAIT")

        self.fsm.add_transition(
            "got_X", "WAIT_DATA", self.x_overheard, "WAIT_DATA")
        self.fsm.add_transition(
            "backoff", "WAIT_DATA", self.x_overheard, "BACKOFF")

        # Transitions from WAIT_ACK
        self.fsm.add_transition(
            "send_DATA", "WAIT_ACK", self.queue_data, "WAIT_ACK")
        self.fsm.add_transition(
            "got_ACK", "WAIT_ACK", self.on_transmit_success, "READY_WAIT")
        self.fsm.add_transition(
            "got_RTS", "WAIT_ACK", self.ignore_rts, "WAIT_ACK")
        self.fsm.add_transition(
            "got_CTS", "WAIT_ACK", self.ignore_cts, "WAIT_ACK")
        self.fsm.add_transition(
            "timeout", "WAIT_ACK", self.on_ack_timeout, "READY_WAIT")

        self.fsm.add_transition(
            "got_X", "WAIT_ACK", self.x_overheard, "WAIT_ACK")
        self.fsm.add_transition(
            "backoff", "WAIT_ACK", self.x_overheard, "BACKOFF")

        # Transitions from BACKOFF
        self.fsm.add_transition("send_DATA", "BACKOFF", self.queue_data, "BACKOFF")
        self.fsm.add_transition("got_RTS", "BACKOFF", self.process_rts, "BACKOFF")
        # self.fsm.add_transition("got_CTS", "BACKOFF", self.ignore_cts,"BACKOFF")  ### This line is more important that what it seems: if we ignore it, we tend to defer all transmissions.
        self.fsm.add_transition("got_CTS", "BACKOFF", self.transmit,
                                "WAIT_ACK")  # This line is more important that what it seems: if we Accept it, we tend to make all transmissions.
        self.fsm.add_transition("got_DATA", "BACKOFF", self.check_pending_data, "BACKOFF")
        # Be careful with this
        self.fsm.add_transition("got_ACK", "BACKOFF", self.check_pending_ack, "READY_WAIT")

        self.fsm.add_transition("got_X", "BACKOFF", self.x_overheard, "BACKOFF")
        self.fsm.add_transition("timeout", "BACKOFF", self.on_timeout, "READY_WAIT")
        self.fsm.add_transition("accept", "BACKOFF", self.send_cts, "WAIT_DATA")

    def x_overheard(self):
        """ By overhearing the channel, we can obtain valuable information.
        If I'm in any state
        """
        if self.fsm.current_state == "READY_WAIT":
            if DEBUG:
                self.logger.debug(
                    "I'm in READY_WAIT and have received a " + self.incoming_packet["type"])

        elif self.fsm.current_state == "WAIT_CTS":
            if DEBUG:
                self.logger.debug(
                    "I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
            self.pending_ack_packet_from = self.last_data_to
            self.fsm.process("backoff")
            return False

        if self.fsm.current_state == "WAIT_DATA":
            if DEBUG:
                self.logger.debug(
                    "I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
            self.pending_data_packet_from = self.last_cts_to
            self.fsm.process("backoff")
            return False

        elif self.fsm.current_state == "WAIT_ACK":
            if DEBUG:
                self.logger.debug(
                    "I'm in " + self.fsm.current_state + " and have overheard " + self.incoming_packet["type"])
            self.pending_ack_packet_from = self.last_data_to
            self.fsm.process("backoff")
            return False

        self.last_packet = self.incoming_packet

        if isinstance(self.layercake.net, RoutingLayer.DSDV) and self.incoming_packet['type'] == "DATA":
            # Can infer routing information from overheard packet
            self.layercake.net.process_update_from_packet(self.incoming_packet)

        t = self.layercake.phy.level2delay(self.incoming_packet['level'])

        backoff = self.get_backoff_time(self.incoming_packet["type"], t)

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.next_timeout > Sim.now() + backoff:
            # I will wait for the longest time
            backoff = self.next_timeout - Sim.now()

        if DEBUG:
            self.logger.debug("Sleep for " + str(backoff) + " due to " + self.incoming_packet["type"]
                              + " coming from " +
                              self.incoming_packet["source"]
                              + " to " + self.incoming_packet["dest"]
                              + " through " + self.incoming_packet["through"])

        self.TimerRequest.signal((backoff, "timeout"))
        self.next_timeout = Sim.now() + backoff

        self.incoming_packet = None

    def ignore_rts(self):
        """


        """
        self.logger.debug("Ignoring RTS received from {src} as I'm currently in {state}".format(
            src=self.incoming_packet["source"],
            state=self.fsm.current_state))
        self.incoming_packet = None

    def ignore_cts(self):
        """


        """
        self.logger.debug("Ignoring CTS received from {src} as I'm waiting on an ACK".format(
            src=self.incoming_packet["source"]))
        self.incoming_packet = None

    def ignore_ack(self):
        """


        """
        self.logger.debug(
            "Ignoring ACK coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def check_pending_data(self):
        """


        """
        if self.pending_data_packet_from is not None:
            if self.incoming_packet["route"][-1][0] == self.pending_data_packet_from:
                self.logger.debug(
                    "Despite everything, I properly received the DATA packet from: " + self.incoming_packet["source"])
                self.on_data_reception()
                self.pending_data_packet_from = None
        else:
            self.logger.warn("I think I have pending data from {pending} but I got something from {src}".format(
                pending=self.pending_data_packet_from,
                src=self.incoming_packet)
            )


    def check_pending_ack(self):
        """


        """
        if self.pending_ack_packet_from is not None:
            if self.incoming_packet["source"] == self.pending_ack_packet_from \
                    or self.pending_ack_packet_from[0:3] == "ANY":
                self.logger.error(
                    "Rescued Pending ACK from {}: {}-{}".format(
                        self.pending_ack_packet_from,
                        self.incoming_packet["source"],
                        self.incoming_packet["ID"])
                )
                self.on_transmit_success()
                self.pending_ack_packet_from = None
        else:
            self.logger.warn("I think I have pending ACK from {src}:{id} but I don't".format(
                src=self.pending_ack_packet_from,
                id=self.last_data_id)
            )

    def send_rts(self):
        """ Request To Send.
        """
        self.last_outgoing_id = self.outgoing_packet_queue[0]['ID']
        self.level = self.outgoing_packet_queue[0]["level"]
        try:
            self.T = self.layercake.phy.level2delay(self.level)
        except:
            self.logger.error("Died trying to get delay for packet {}".format(
                self.outgoing_packet_queue[0]
            ))
            raise

        if self.layercake.phy.is_idle():
            self.total_channel_access_attempts += 1
            self.transmission_attempts += 1
            self.channel_access_retries = 0

            if self.outgoing_packet_queue[0]["through"].startswith("ANY"):
                self.multicast = True
            else:
                self.multicast = False

            rts_packet = {"type": "RTS", "ID": self.outgoing_packet_queue[0]["ID"],
                          "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                          "dest": self.outgoing_packet_queue[0]["dest"],
                          "dest_position": self.outgoing_packet_queue[0]["dest_position"],
                          "through": self.outgoing_packet_queue[0]["through"],
                          "through_position": self.outgoing_packet_queue[0]["through_position"],
                          "length": self.rts_packet_length, "level": self.outgoing_packet_queue[0]["level"],
                          "time_stamp": Sim.now()}

            self.last_data_to = self.outgoing_packet_queue[0]["through"]
            self.level = self.outgoing_packet_queue[0]["level"]
            self.T = self.layercake.phy.level2delay(self.level)

            self.layercake.phy.transmit_packet(rts_packet)

            timeout = self.get_timeout("CTS", self.T) + random.random() * (self.transmission_attempts / 4.0)
            if DEBUG:
                self.logger.debug("Transmitting RTS to {dest} for {id} and waiting {timeout}".format(
                    dest=self.outgoing_packet_queue[0]["dest"],
                    id=self.outgoing_packet_queue[0]["ID"],
                    timeout=timeout
                ))
            self.TimerRequest.signal((timeout, "timeout"))
            self.fsm.current_state = "WAIT_CTS"

            self.time = Sim.now()
            self.next_timeout = self.time + timeout

        # I'm currently not limiting the number of channel access retries
        else:
            self.channel_access_retries += 1
            timeout = random.random() * (2 * self.T + self.t_data)

            self.TimerRequest.signal((timeout, self.fsm.input_symbol))

            if DEBUG:
                self.logger.debug("Channel not clear to transmit {id} to {dest} via {thru}, backing off for {t}".format(
                    id=self.outgoing_packet_queue[0]['ID'],
                    dest=self.outgoing_packet_queue[0]['dest'],
                    thru=self.outgoing_packet_queue[0]['through'],
                    t=timeout
                ))

            self.time = Sim.now()
            self.next_timeout = self.time + timeout

    def process_rts(self):
        """ Maybe I can do it now.
        """
        if self.last_packet["type"] == "CTS" and self.last_packet["through"] == self.incoming_packet["source"]:
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")
        elif self.last_packet["type"] == "DATA" and self.last_packet["through"] == self.incoming_packet["source"] and \
                        self.last_packet["dest"] == self.incoming_packet["dest"]:
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("accept")

    def get_backoff_time(self, packet_type, t):
        """ Returns the backoff for a specific state.
        :param packet_type:
        :param t:
        """
        if packet_type == "RTS":
            backoff = 4 * t + 2 * self.t_control + self.t_data
        elif packet_type == "CTS" or packet_type == "SIL":
            backoff = 3 * t + self.t_control + self.t_data
        elif packet_type == "DATA":
            backoff = 2 * t + self.t_control
        elif packet_type == "ACK":
            backoff = 0  # I'm all set

        if DEBUG > 1:
            self.logger.debug("Backoff: {t} based on {t} {pkt}".format(
                t=backoff, T=t, pkt=packet_type

            ))
        return backoff

    def get_timeout(self, packet_type, t):
        """ Returns the timeout for a specific state.
        :param packet_type:
        :param t:
        """

        if packet_type == "CTS":
            t = 2 * t + 2 * self.t_control
        elif packet_type == "DATA":
            t = 2 * t + self.t_data + self.t_control
        elif packet_type == "ACK":
            t = 2 * t + self.t_data + self.t_control

        if DEBUG > 1:
            self.logger.debug("Timeout: {t} based on {t} {pkt}".format(
                t=t, T=t, pkt=packet_type
            ))
        return t

    def process_cts(self):
        """ The CTS has been received.
        """
        self.logger.debug("CTS from " + self.incoming_packet[
            "source"] + " coming from " + self.incoming_packet["through"] + " properly received.")

        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        self.incoming_packet = None

    def check_rts(self):
        """ Before proceeding with the data transmission, just check if it's a duplicated packet (maybe the ACK collided).
        """
        if self.layercake.net.have_duplicate_packet(self.incoming_packet):
            self.send_ack(self.incoming_packet["source"])
        else:
            self.send_cts()
            self.fsm.current_state = "WAIT_DATA"

    def send_cts(self):
        """ Clear To Send: I'm proposing myself as a good candidate for the next transmission or I just let transmit if I have been
        already selected.
        """
        # Update the timer: 1.-Stop, 2.-Restart
        p = Sim.Process()
        p.interrupt(self.timer)

        if self.incoming_packet["through"].startswith("ANY"):
            self.multicast = True
        else:
            self.multicast = False

        cts_packet = {"type": "CTS", "ID": self.incoming_packet["ID"],
                      "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                      "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                      "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                      "length": self.cts_packet_length, "rx_energy": self.layercake.phy.rx_energy,
                      "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

        if DEBUG:
            self.logger.debug("Transmitting CTS to {src} in response to {id}".format(
                src=self.incoming_packet["source"],
                id=self.incoming_packet["ID"],
            )
            )
        self.last_cts_to = self.incoming_packet["source"]
        self.last_cts_from = self.incoming_packet["dest"]

        self.layercake.phy.transmit_packet(cts_packet)

        self.level = self.incoming_packet["level"]
        self.T = self.layercake.phy.level2delay(self.level)

        timeout = self.get_timeout("DATA", self.T) + random.random() * (self.channel_access_retries + 0.01) / 4.0
        self.TimerRequest.signal((timeout, "timeout"))

        self.time = Sim.now()
        self.next_timeout = self.time + timeout

        self.pending_data_packet_from = self.incoming_packet["source"]
        self.incoming_packet = None

    def send_ack(self, packet_origin):
        """ Sometimes we can not use implicit ACKs.
        :param packet_origin:
        """
        if DEBUG:
            self.logger.info("ACK to {} in response to {}".format(
                packet_origin, self.incoming_packet["ID"])
            )

        # This may be incorrect
        if self.incoming_packet["through"].startswith("ANY"):
            self.multicast = True
        else:
            self.multicast = False

        ack_packet = {"type": "ACK",
                      "source": self.layercake.hostname, "source_position": self.layercake.get_current_position(),
                      "dest": packet_origin, "dest_position": None,
                      "through": packet_origin, "through_position": None,
                      "length": self.ack_packet_length, "level": self.incoming_packet["level"],
                      "ID": self.incoming_packet["ID"]}

        self.layercake.phy.transmit_packet(ack_packet)

    def initiate_transmission(self, outgoing_packet):
        """ Function called from the upper layers to transmit a packet.
        :param outgoing_packet:
        """
        if DEBUG:
            self.logger.debug("Prepping {id} for launch to {dest}".format(
                id=outgoing_packet["ID"],
                dest=outgoing_packet["dest"]
            )
            )
        self.outgoing_packet_queue.append(outgoing_packet)
        self.fsm.process("send_DATA")

    def on_new_packet_received(self, incoming_packet):
        """ Function called from the lower layers when a packet is received.
        :param incoming_packet:
        """
        self.incoming_packet = incoming_packet
        if self.can_i_help(incoming_packet):
            self.fsm.process(self.packet_signal[self.incoming_packet["type"]])
        else:
            self.overhearing()

    def can_i_help(self, packet):
        """ A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        :param packet:
        """
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

        return False

    def overhearing(self):
        """ Valuable information can be obtained from overhearing the channel.
        """
        if self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK" \
                and self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet[
            "dest"] == self.outgoing_packet_queue[0]["dest"]:
            # This is an implicit ACK
            self.fsm.process("got_ACK")
        else:
            self.fsm.process("got_X")

    def on_data_reception(self):
        """ After the RTS/CTS exchange, a data packet is received. We should acknowledge the previous node or we can try to use implicit
        acknowledges if the that does not mean a waste of power.
        """
        if self.fsm.current_state != "READY_WAIT":
            # We got a DATA packet, we should stop the timer.
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.current_state = "READY_WAIT"

        if self.fsm.current_state != "BACKOFF":
            self.last_packet = None

        # We should acknowledge the previous node, this is not an end-to-end
        # ACK
        packet_origin = self.incoming_packet["route"][-1][0]

        try:
            if self.layercake.net.need_explicit_ack(self.incoming_packet["level"], self.incoming_packet["dest"]) or len(
                    self.outgoing_packet_queue) != 0:
                self.send_ack(packet_origin)
        except KeyError:
            self.logger.error("Fucked up on {}, net had {}".format(
                self.incoming_packet, self.layercake.net.items()))
            raise

        self.layercake.net.on_packet_reception(self.incoming_packet)

        self.incoming_packet = None

    def on_error(self):
        """ This function is called when the FSM has an unexpected input_symbol in a determined state.
        """
        if DEBUG:
            self.logger.error(
                "TRANS {sm.last_state}:{sm.last_symbol}:{sm.last_action} to {sm.current_state} due to {sm.input_symbol}: \n{src} {thru} {dest} {id} {type}".format(
                    sm=self.fsm,
                    src=self.incoming_packet['source'],
                    thru=self.incoming_packet.get('through'),
                    dest=self.incoming_packet['dest'],
                    id=self.incoming_packet.get('ID'),
                    type=self.incoming_packet['type']
                )
            )
            pass

    def on_transmit_success(self):
        """ When an ACK is received, we can assume that everything has gone fine, so it's all done.
        """
        # We got an ACK, we should stop the timer.
        p = Sim.Process()
        p.interrupt(self.timer)

        self.layercake.signal_good_tx(self.incoming_packet['ID'])  # ACKd packets should always have IDs
        self.post_success_or_fail()

    def on_transmit_fail(self):
        """ All the transmission attempts have been completed. It's impossible to reach the node.
        """
        self.layercake.signal_lost_tx(self.outgoing_packet_queue[0]['ID'])
        self.logger.warn("Failed to transmit to {}".format(
            self.outgoing_packet_queue[0]["dest"]
        ))
        self.post_success_or_fail()

    def post_success_or_fail(self):
        """ Successfully or not, we have finished the current transmission.
        """
        try:
            self.outgoing_packet_queue.pop(0)["dest"]
        except IndexError as e:
            self.logger.fatal("Over Popped: {sm.current_state}".format(
                sm=self.fsm
            ))
            raise
        self.transmission_attempts = 0

        # Is there anything else to do?
        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0

    def on_ack_timeout(self):
        """ The timeout has expired and NO ACK has been received.
        """
        self.transmission_attempts += 1

        if self.transmission_attempts < self.max_transmission_attempts:
            random_delay = self.transmission_attempts + random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))

    def on_data_timeout(self):
        """ The timeout has expired and NO DATA has been received.
        """
        self.logger.debug(
            "Timed Out!, No Data Received for {}".format(self.last_cts_to))

        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0

    def on_cts_timeout(self):
        """


        """
        self.transmission_attempts += 1
        if DEBUG:
            self.logger.debug("CTS timeout waiting on {}: Attempt {}".format(
                self.outgoing_packet_queue[0]['ID'],
                self.transmission_attempts)
            )
        if self.layercake.phy.collision_detected():
            self.logger.debug("It seems that there has been a collision.")

        if self.transmission_attempts > self.max_transmission_attempts:
            self.logger.warn("CTS timeout limit waiting on {} after {}".format(
                self.outgoing_packet_queue[0]['ID'],
                self.transmission_attempts)
            )
        else:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))

    def on_timeout(self):

        """


        """
        if len(self.outgoing_packet_queue) > 0:
            random_delay = random.random() * self.max_wait_to_retransmit
            self.TimerRequest.signal((random_delay, "send_DATA"))
            self.transmission_attempts = 0

    def transmit(self):
        """


        """
        p = Sim.Process()
        p.interrupt(self.timer)

        self.transmitnotimer()

    def transmitnotimer(self):
        """


        """
        self.layercake.phy.transmit_packet(self.outgoing_packet_queue[0])
        self.last_data_to = self.outgoing_packet_queue[0]["through"]
        self.last_data_id = self.outgoing_packet_queue[0]["ID"]
        timeout = self.get_timeout("ACK", self.T)

        if DEBUG:
            self.logger.debug("DATA/ACK Timeout for {}:{} to {} for {}".format(
                self.outgoing_packet_queue[0]['type'],
                self.outgoing_packet_queue[0]['ID'],
                self.outgoing_packet_queue[0]['dest'],
                timeout))
        self.TimerRequest.signal((timeout, "timeout"))

    def queue_data(self):
        """


        """
        if DEBUG:
            self.logger.info("Queuing Data to {}:{} as we are currently {}".format(
                self.outgoing_packet_queue[0]['dest'],
                self.outgoing_packet_queue[0]['ID'],
                self.fsm.current_state

            ))


class FAMA(CSMA):
    """
    Floor Acquisition Multiple Access

    Should be slotted but is not currently
    """

    def get_backoff_time(self, packet_type, t):
        """ Returns the backoff for a specific state.
        :param packet_type:
        :param t:
        t : Maximum Propogation Delay
        """
        if packet_type == "RTS":
            backoff = 4 * t
        elif packet_type == "CTS" or packet_type == "SIL":
            backoff = 3 * t
        elif packet_type == "DATA":
            backoff = 2 * t
        elif packet_type == "ACK":
            backoff = 0  # I'm all set

        if DEBUG:
            self.logger.debug("Backoff: {t} based on {t} {pkt}".format(
                t=backoff, T=t, pkt=packet_type

            ))
        return backoff

    def get_timeout(self, packet_type, t):
        """ Returns the timeout for a specific state.
        :param packet_type:
        :param t:
        t : Maximum Propogation Delay
        """

        if packet_type == "CTS":
            t = 2 * t + 2 * self.t_control
        elif packet_type == "DATA":
            t = 2 * t + self.t_data + self.t_control
        elif packet_type == "ACK":
            t = 2 * t + self.t_data + self.t_control

        if DEBUG:
            self.logger.debug("Timeout: {t} based on {t} {pkt}".format(
                t=t, T=t, pkt=packet_type
            ))
        return t


class CSMA4FBR(CSMA):
    """CSMA for FBR: adaptation of CSMA with power control to couple it with FBR
    """

    def __init__(self, node, config):

        CSMA.__init__(self, node, config)

        # A new packet type
        self.packet_signal["SIL"] = "got_SIL"
        self.SIL_packet_length = None

        self.valid_candidates = {}
        self.original_through = None

        # By MC_RTS we refer to MultiCast RTS packets. We should consider to
        # process them.
        self.fsm.add_transition("got_MC_RTS", "READY_WAIT", self.process_mc_rts, "AWARE")
        self.fsm.add_transition("got_SIL", "READY_WAIT", self.ignore_sil, "READY_WAIT")

        # Definition of a new state, AWARE, and its transitions
        self.fsm.add_transition("send_CTS", "AWARE", self.send_cts, "WAIT_DATA")
        self.fsm.add_transition("ignore_RTS", "AWARE", self.x_overheard, "BACKOFF")

        # Now I am receiving several CTS, then, I should just append them
        # Maybe I should tell him to be silent
        self.fsm.add_transition("got_MC_RTS", "WAIT_CTS", self.process_mc_rts, "WAIT_CTS")
        self.fsm.add_transition("got_CTS", "WAIT_CTS", self.append_cts, "WAIT_CTS")
        self.fsm.add_transition("got_SIL", "WAIT_CTS", self.process_sil, "BACKOFF")

        self.fsm.add_transition("timeout", "WAIT_CTS", self.select_cts, "WAIT_CTS")
        self.fsm.add_transition("transmit", "WAIT_CTS", self.transmit, "WAIT_ACK")
        self.fsm.add_transition("retransmit", "WAIT_CTS", self.send_rts, "WAIT_CTS")
        self.fsm.add_transition("abort", "WAIT_CTS", self.on_transmit_fail, "READY_WAIT")
        self.fsm.add_transition("defer", "WAIT_CTS", self.x_overheard, "BACKOFF")
        self.fsm.add_transition("got_DATA", "WAIT_CTS", self.check_pending_data(), "READY_WAIT")

        # New transition from WAIT_DATA: I have been not selected as the next
        # hop
        self.fsm.add_transition("got_MC_RTS", "WAIT_DATA", self.process_mc_rts, "WAIT_DATA")
        self.fsm.add_transition("ignored", "WAIT_DATA", self.x_overheard, "BACKOFF")
        self.fsm.add_transition("got_CTS", "WAIT_DATA", self.ignore_cts, "WAIT_DATA")
        self.fsm.add_transition("got_SIL", "WAIT_DATA", self.ignore_sil, "WAIT_DATA")

        self.fsm.add_transition("got_MC_RTS", "BACKOFF", self.process_mc_rts, "BACKOFF")
        self.fsm.add_transition("got_SIL", "BACKOFF", self.ignore_sil, "BACKOFF")
        self.fsm.add_transition("defer", "BACKOFF", self.x_overheard, "BACKOFF")

        self.fsm.add_transition("got_MC_RTS", "WAIT_ACK", self.process_mc_rts, "WAIT_ACK")
        self.fsm.add_transition("got_SIL", "WAIT_ACK", self.ignore_sil, "WAIT_ACK")

    def activate(self):
        self.SIL_packet_length = self.config["sil_packet_length"]
        super(CSMA4FBR, self).activate()

    def ignore_sil(self):
        """


        """
        assert self.incoming_packet['type'] == "SIL", "Ignoring a non-SIL packet: {}".format(
            self.incoming_packet
        )
        if DEBUG:
            self.logger.info(
                "Ignoring SIL coming from " + self.incoming_packet["through"])
        self.incoming_packet = None

    def process_mc_rts(self):
        """ Someone is looking for help, may I help? Now the active nodes are the ones that transmit, maybe we should do it the opposite way
        """
        if self.fsm.current_state == "AWARE":
            if self.layercake.net.i_am_a_valid_candidate(self.incoming_packet):
                if self.layercake.net.have_duplicate_packet(self.incoming_packet):
                    self.send_ack(self.incoming_packet["source"])
                    self.fsm.current_state = "READY_WAIT"
                else:
                    self.multicast = True
                    self.fsm.process("send_CTS")
            else:
                self.logger.debug("I can't attend the MultiCast RTS received from " + self.incoming_packet[
                    "source"] + " but I will be silent.")
                self.fsm.process("ignore_RTS")
        elif self.fsm.current_state == "WAIT_DATA":
            if self.last_cts_to == self.incoming_packet["source"]:
                self.ignore_rts()  # Think of this
                # The silence part is being revised.
                # else:
                # self.send_silence()
                # elif self.fsm.current_state == "WAIT_CTS" or self.fsm.current_state == "WAIT_TIME":
                # self.send_silence()
                # elif self.fsm.current_state == "BACKOFF":
                # self.send_silence()

    def send_silence(self):
        """ Please be quiet!
        """
        sil_packet = {"type": "SIL",
                      "source": self.incoming_packet["dest"], "source_position": self.incoming_packet["dest_position"],
                      "dest": self.incoming_packet["source"], "dest_position": self.incoming_packet["source_position"],
                      "through": self.layercake.hostname, "through_position": self.layercake.get_current_position(),
                      "length": self.SIL_packet_length, "tx_energy": self.layercake.phy.tx_energy,
                      "time_stamp": self.incoming_packet["time_stamp"], "level": self.incoming_packet["level"]}

        self.logger.debug(
            "Transmitting SIL to " + self.incoming_packet["source"])

        self.layercake.phy.transmit_packet(sil_packet)

        self.incoming_packet = None

    def process_sil(self):
        """ The next time that I try to find a route I should do it starting again from ANY0.
        """
        self.outgoing_packet_queue[0]["through"] = "ANY0"
        self.outgoing_packet_queue[0]["level"] = 0
        self.x_overheard()

    def append_cts(self):
        """ More than one CTS is received when looking for the next best hop. We should consider all of them.
        The routing layer decides.
        """
        self.valid_candidates[self.incoming_packet["through"]] = \
            (Sim.now() - self.incoming_packet["time_stamp"]) / 2.0, \
            self.incoming_packet["rx_energy"], \
            self.incoming_packet["through_position"]
        if DEBUG > 1:
            self.logger.debug("Appending CTS to {src} coming from {thru}@{pos}".format(
                src=self.incoming_packet["source"],
                thru=self.incoming_packet["through"],
                pos=self.incoming_packet['through_position'])
            )
        self.layercake.net.add_node(
            self.incoming_packet["through"], self.incoming_packet["through_position"])

        self.incoming_packet = None

        if not self.multicast:
            # Update the timer: 1.-Stop, 2.-Restart
            p = Sim.Process()
            p.interrupt(self.timer)
            self.fsm.process("timeout")

    def select_cts(self):
        """ Once we have wait enough, that is, 2 times the distance at which the best next hop should be, we should select it.
        """
        current_through = self.outgoing_packet_queue[0]["through"]
        self.outgoing_packet_queue[0]["through"], self.outgoing_packet_queue[0][
            "through_position"] = self.layercake.net.select_route(
            self.valid_candidates, self.outgoing_packet_queue[0]["through"], self.transmission_attempts,
            self.outgoing_packet_queue[0]["dest"])

        if self.outgoing_packet_queue[0]["through"] == "ABORT":
            # We have consumed all the attempts
            if DEBUG:
                self.logger.warn("Aborting {type}-{id} to {dest}".format(
                    type=self.outgoing_packet_queue[0]['type'],
                    id=self.outgoing_packet_queue[0]['ID'],
                    dest=self.outgoing_packet_queue[0]['dest'],
                ))
            self.fsm.process("abort")

        elif self.outgoing_packet_queue[0]["through"][0:3] == "ANY":
            # We should retransmit increasing the power
            self.outgoing_packet_queue[0]["level"] = int(
                self.outgoing_packet_queue[0]["through"][3])
            if DEBUG:
                self.logger.debug("Raising {type}-{id} to {dest} to tx level {level}".format(
                    type=self.outgoing_packet_queue[0]['type'],
                    id=self.outgoing_packet_queue[0]['ID'],
                    dest=self.outgoing_packet_queue[0]['dest'],
                    level=self.outgoing_packet_queue[0]['level']
                ))
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"] == "2CHANCE":
            # We should give a second chance to current node
            self.outgoing_packet_queue[0]["through"] = current_through
            if DEBUG:
                self.logger.debug("Second Chance to {type}-{id} to {dest} via {through}".format(
                    type=self.outgoing_packet_queue[0]['type'],
                    id=self.outgoing_packet_queue[0]['ID'],
                    dest=self.outgoing_packet_queue[0]['dest'],
                    through=self.outgoing_packet_queue[0]['through'],
                ))
            self.fsm.process("retransmit")

        elif self.outgoing_packet_queue[0]["through"][0:5] == "NEIGH":
            # With current transmission power level, the destination has become
            # a neighbor
            self.outgoing_packet_queue[0][
                "through"] = self.outgoing_packet_queue[0]["dest"]
            self.outgoing_packet_queue[0][
                "through_position"] = self.outgoing_packet_queue[0]["dest_position"]

            if DEBUG:
                self.logger.debug("Destination of {type}-{id} to {dest} is neighbour {dist} away".format(
                    type=self.outgoing_packet_queue[0]['type'],
                    id=self.outgoing_packet_queue[0]['ID'],
                    dest=self.outgoing_packet_queue[0]['dest'],
                    dist=distance(self.outgoing_packet_queue[0][
                                      'dest_position'], self.layercake.get_current_position()),
                ))
            self.fsm.process("retransmit")

        else:
            self.fsm.process("transmit")

    def transmit(self):
        """ Real Transmission of the Packet.
        """
        try:
            self.layercake.net.process_update_from_packet(
                self.outgoing_packet_queue[0]["dest"],
                self.outgoing_packet_queue[0]["dest_position"],
                self.outgoing_packet_queue[0]["through"],
                self.outgoing_packet_queue[0]["through_position"]
            )
        except IndexError:
            self.logger.error(
                "Problem with the queue on trying to transmit: {}".format(self.outgoing_packet_queue))
            raise
        CSMA.transmitnotimer(self)

    def get_timeout(self, packet_type, t):
        """ Returns the timeout for a specific state.
        :rtype : int
        :param packet_type:
        :param t:
        """

        if packet_type == "CTS":
            t = 2 * t + 2 * self.t_control
        elif packet_type == "DATA":
            if self.multicast:
                t = 3 * t + self.t_data + self.t_control
            elif not self.multicast:
                t = 2 * t + self.t_data + self.t_control
        elif packet_type == "ACK":
            t = 2 * t + self.t_data + self.t_control

        return t

    def overhearing(self):
        """ Valuable information can be obtained from overhearing the channel.
        """
        if self.incoming_packet["type"] == "DATA" and self.fsm.current_state == "WAIT_DATA":
            if self.incoming_packet["route"][-1][0] == self.last_cts_to and self.incoming_packet[
                "through"] == self.last_cts_from:
                # I have not been selected as the next hop. I read it from the
                # data.
                self.fsm.process("ignored")

        elif self.incoming_packet["type"] == "RTS" and self.fsm.current_state == "WAIT_ACK":
            if self.incoming_packet["source"] == self.outgoing_packet_queue[0]["through"] and self.incoming_packet[
                "dest"] == self.outgoing_packet_queue[0]["dest"]:
                # This is an implicit ACK
                self.fsm.process("got_ACK")

        elif self.incoming_packet["type"] == "CTS" and self.fsm.current_state == "WAIT_DATA" and self.last_cts_to == \
                self.incoming_packet["dest"]:
            # This is another candidate proposing himself as a good candidate
            self.fsm.process("got_CTS")

        elif self.incoming_packet["type"] == "SIL":
            return False

        else:
            self.fsm.process("got_X")

    def can_i_help(self, packet):
        """ A node may be able to help within a transmission if the packet is addressed to it or it is a multicast packet.
        :param packet:
        """
        # Is this a packet already "directed" to me? I'm not checking if only
        # dest its me to avoid current protocol errors. Should be revised.
        if packet["through"] == self.layercake.hostname and packet["dest"] == self.layercake.hostname:
            return True
        if packet["through"] == self.layercake.hostname:
            return True

        # CTS, ACK, WAR and SIL are the only types of packet that are directly
        # addressed
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

    def on_new_packet_received(self, incoming_packet):
        """ Function called from the lower layers when a packet is received.
        :param incoming_packet:
        """
        self.incoming_packet = incoming_packet

        if self.can_i_help(self.incoming_packet):
            if self.incoming_packet["through"][0:3] == "ANY":
                self.fsm.process("got_MC_RTS")
            else:
                self.fsm.process(
                    self.packet_signal[self.incoming_packet["type"]])
        else:
            self.overhearing()

    def send_rts(self):
        """ The RTS sent is the normal one, but we should initialize the list of replies.
        """
        self.valid_candidates = {}
        CSMA.send_rts(self)
