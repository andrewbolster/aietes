#!/usr/bin/env python
"""
 * This file is part of the Aietes Framework (https://github.com/andrewbolster/aietes)
 *
 * (C) Copyright 2013 Andrew Bolster (http://andrewbolster.info/) and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

from Packet import MACPacket, ACK
from aietes.Tools import debug, Sim, distance2Bandwidth
from aietes.Tools.FSM import *

from random import random


debug = False


class MAC():

    """Generic Class for MAC Algorithms
    The only real difference between MAC's are their packet types and State Machines
    """

    def __init__(self, layercake, config=None):
        self.logger = layercake.logger.getChild("%s" % self.__class__.__name__)
        self.logger.info('creating instance')
        self.config = config
        self.layercake = layercake
        self.node = self.layercake.host
        self.data_packet_length = config["data_packet_length"]

        self.macBuilder()

        # These need to be configurabled
        self.max_retransmit = 4
        self.retransmit_timeout = 10
        self.ack_timeout = 4

        self.InitialiseStateEngine()

    def activate(self):
        self.logger.info('activating instance')
        self.timer = self.InternalTimer(self.sm)  # Instance of Internal Timer
        # SimPy Event for signalling
        self.timer_event = Sim.SimEvent("timer_event")
        self.channel_access_retries = 0
        self.transmission_attempts = 0
        self.outgoing_queue = []
        self.incoming_packet = None
        # Tie the event to lifecycle function
        Sim.activate(self.timer, self.timer.lifecycle(self.timer_event))

    def macBuilder(self):
        """Generate run-time MAC config
        """
        raise TypeError("Tried to instantiate the base MAC class")

    class InternalTimer(Sim.Process):

        """The internal Timer of the MACs is a SimPy Process as well as the nodes themselves.
        In a practical sence, this mirrors the idea that the network interfaces on the Vectors
        operate independently from the 'Command and Control' functionality
        """

        def __init__(self, state_machine):
            Sim.Process.__init__(
                self, name="%s_Timer" % self.__class__.__name__)
            self.sm = state_machine

        def lifecycle(self, Request):
            while True:
                yield Sim.waitevent, self, Request
                # Wait for a given time
                yield Sim.hold, self, Request.signalparam[0]
                if self.interrupted():
                    self.interruptReset()
                else:
                    self.sm.process(Request.signalparam[1])  # Do something

    def InitialiseStateEngine(self):
        """Generate a FSM with an initial READY_WAIT state
        """
        self.sm = FSM("READY_WAIT", [])
        # Default transition to error state to fallback
        self.sm.set_default_transition(self.onError, "READY_WAIT")

    def send(self, FromAbove):
        """Function Called from upper layers to send a packet
        Encapsulates the Route layer packet in to a MAC Packet
        """
        self.outgoing_queue.append(MACPacket(FromAbove))
        self.sm.process("send_DATA")

    def transmit(self):
        """Real Transmission of packet to physical layer
        On successful channel acquisition, bails out to "WAIT_ACK"
        """
        packet = self.outgoing_queue[0]
        if self.layercake.phy.variable_bandwidth:
            distance = self.layercake.phy.var_power[
                'levelToDistance'][packet.tx_level]
            bandwidth = distance2Bandwidth(self.power,
                                           self.layercake.phy.frequency,
                                           distance,
                                           self.layercake.phy.threshold['SNR']
                                           )
        else:
            bandwidth = self.layercake.phy.bandwidth

        duration = packet.length / \
            self.layercake.phy.bandwidth_to_bit(bandwidth)

        in_air_time = self.layercake.phy.range / \
            self.layercake.phy.medium_speed

        timeout = 2 * duration + 2 * in_air_time

        if self.layercake.phy.isIdle():
            self.transmission_attempts += 1
            self.channel_access_retries = 0
            self.layercake.phy.send(packet)
            if packet.requests_ack:
                self.sm.current_state = "WAIT_ACK"
                self.logger.info("Waiting on ACK for {}".format(packet))
                self.timer_event.signal((timeout, "timeout"))
            else:
                self.sm.current_state = "READY_WAIT"
                self.logger.info("NOT Waiting on ACK for {}".format(packet))
        else:
            self.logger.info("Channel Not Idle")
            self.channel_access_retries += 1
            timeout = random() * (2 * in_air_time)
            self.timer_event.signal((timeout, self.sm.input_symbol))

    def onTX_success(self):
        """When an ACK has been received, we can assume it all went well
        """
        Sim.Process().interrupt(self.timer)
        self.logger.debug("Got Ack from {incoming.source}({out.next_hop}: {incoming.data})".format(
            out=self.outgoing_queue[0],
            incoming=self.incoming_packet
        ))
        self.postTX()

    def onTX_fail(self):
        """When an ACK has timedout, we can assume it is impossible to contact the next hop
        """
        self.logger.warn("Timed out TX to " + self.outgoing_queue[0].next_hop)
        self.postTX()

    def postTX(self):
        """Succeeded or given up, either way, tidy up
        """
        self.logger.debug(
            "Tidying up packet to " + self.outgoing_queue[0].next_hop)
        self.outgoing_queue.pop(0)
        self.transmission_attempts = 0

        if len(self.outgoing_queue) > 0:
            random_delay = random() * self.retransmit_timeout
            self.timer_event.signal((random_delay, "send_DATA"))

    def recv(self, FromBelow):
        """Function Called from lower layers to recieve a packet
        Decapsulates the packet from the physical
        equiv to OnNewPacket
        """
        self.incoming_packet = FromBelow.decap()
        if FromBelow.isFor(self.node.name):
            if debug:
                self.logger.debug("Processing {} from {}".format(
                    self.signals[FromBelow.type], FromBelow.payload))
            self.sm.process(self.signals[FromBelow.type])
        else:
            self.overheard()

    def overheard(self):

        pass

    def onRX(self):
        """received a packet
        Should ack, but can drop ack if routing layer says its ok
        Sends packet up to higher level
        """
        origin = self.incoming_packet.last_sender()

        if self.layercake.net.explicitACK(self.incoming_packet):
            self.layercake.phy.send(ACK(self.node, self.incoming_packet))
        # Send up to next level in stack
        self.layercake.net.recv(self.incoming_packet)

    def onError(self):
        """ Called via state machine when unexpected input symbol in a determined state

        This is usually caused by packet duplication in the air or broadcast destination addresses with multiple nodes
        """
        self.logger.error(
            "Unexpected transition from {sm.current_state} because of symbol {sm.input_symbol} with Packet {pkt}".format(
                sm=self.sm,
                pkt=self.incoming_packet
            )
        )

    def onTimeout(self):
        """When it all goes wrong
        """
        self.transmission_attempts += 1
        self.logger.info("ACK Timeout")
        if self.transmission_attempts > self.max_retransmit:
            self.sm.process("fail")
        else:
            self.timer_event.signal(
                (random() * self.retransmit_timeout, "resend"))

    def queueData(self):
        """Log queueing
        """  # TODO what the hell is this?
        self.logger.info("Queueing Data")

    def draw(self):
        import pydot

        graph = pydot.Dot(graph_type='digraph')

        # create base state nodes (set of destination states)
        basestyle = ''
        states = {}
        for state in set(zip(*tuple(self.sm.state_transitions.values()))[1]):
            states[state] = pydot.Node(state, *basestyle)
            graph.add_node(states[state])

        # create function 'nodes'
        functions = {}
        for function in set(zip(*tuple(self.sm.state_transitions.values()))[0]):
            functions[function.__name__] = pydot.Node(
                function.__name__, shape="parallelogram")
            graph.add_node(functions[function.__name__])

        for (signal, state) in self.sm.state_transitions:
            (function, next_state) = self.sm.state_transitions[(signal, state)]
            graph.add_edge(
                pydot.Edge(states[state], functions[function.__name__], label=signal))
            # differently formatted return edge
            graph.add_edge(
                pydot.Edge(functions[function.__name__], states[next_state], style='dotted', shape='onormal'))

        graph.write_png("%s.png" % self.__class__.__name__)


class ALOHA(MAC):

    """A very simple algorithm
    """

    def macBuilder(self):
        self.signals = {
            'ACK': "got_ACK",
            'DATA': "got_DATA"
        }

        # Adapted/derived variables
        self.timeout = 0  # co-adapted with TX pwr
        self.level = 0  # derived from routing layer
        self.T = 0

    def InitialiseStateEngine(self):
        """Set up the state machine for ALOHA
        """
        MAC.InitialiseStateEngine(self)
        ##############################
        # Transitions from READY_WAIT
        self.sm.add_transition(
            "got_DATA", "READY_WAIT", self.onRX, "READY_WAIT")
        self.sm.add_transition(
            "send_DATA", "READY_WAIT", self.transmit, "READY_WAIT")

        ##############################
        # Transitions from WAIT_ACK
        self.sm.add_transition("got_DATA", "WAIT_ACK", self.onRX, "WAIT_ACK")
        self.sm.add_transition(
            "send_DATA", "WAIT_ACK", self.queueData, "WAIT_ACK")
        self.sm.add_transition(
            "got_ACK", "WAIT_ACK", self.onTX_success, "READY_WAIT")
        self.sm.add_transition(
            "timeout", "WAIT_ACK", self.onTimeout, "WAIT_2_RESEND")

        ##############################
        # Transitions from WAIT_2_RESEND
        self.sm.add_transition(
            "resend", "WAIT_2_RESEND", self.transmit, "WAIT_2_RESEND")
        self.sm.add_transition(
            "got_DATA", "WAIT_2_RESEND", self.onRX, "WAIT_2_RESEND")
        self.sm.add_transition(
            "send_DATA", "WAIT_2_RESEND", self.queueData, "WAIT_2_RESEND")
        self.sm.add_transition(
            "fail", "WAIT_2_RESEND", self.onTX_fail, "READY_WAIT")

    def overheard(self):
        """Steal some information from overheard packets
        i.e. implicit ACK: Source hears forwarding transmission of its own packet
        """
        if self.incoming_packet.type == "DATA" and self.sm.current_state == "WAIT_ACK":
            last_hop = self.incoming_packet.route[-1]['name']
            if self.outgoing_queue[0].next_hop == last_hop and self.outgoing_queue[0].id == self.incoming_packet.id:
                self.logger.debug("received an implicit ACK from routing node {lh}: Data={packet.data}".format(
                    lh=last_hop,
                    packet=self.incoming_packet
                ))
                self.sm.process("got_ACK")
                # TODO Expand/Duplicate with overhearing position information
