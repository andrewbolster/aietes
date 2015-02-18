#!/usr/bin/env python
# coding=utf-8
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

import PhysicalLayer as Phy
import MAC
import RoutingLayer as Net


class Layercake(object):
    """
    Defines the Four relevant network layers for a given node
    Phy,MAC,Network,Application
    """

    def __init__(self, host, config):
        """ Generate a Layercake with the stated config dict

        Args:
            host (Node):    This layercake's master node
            config (dict):  Layer-specific config settings

        Returns:
            A Layercake instance

        """

        self.monitor_mode = None
        self.app_rx_handler = None
        self.host = host
        self.hostname = host.name
        self.config = config
        self.channel_event = self.host.simulation.channel_event
        self.logger = host.logger.getChild("%s" % self.__class__.__name__)
        self.sim_duration = host.simulation.duration_intervals
        self.packet_length = None
        ###
        # App Signal Handlers
        ###
        self.tx_good_signal_hdlrs = []
        self.tx_lost_signal_hdlrs = []
        self.fwd_signal_hdlrs = []
        self.pwd_signal_hdlr = None

        ##############################
        # Phy
        ##############################
        try:
            phy_mod = getattr(Phy, str(config['phy']))
            self.phy = phy_mod(self,
                               self.config['PHY'],
                               self.channel_event)
        # except AttributeError:
        # raise ConfigError("Can't find Phy: %s" % config['Phy'])
        #
        except:
            raise

        ##############################
        # MAC
        ##############################
        try:
            mac_mod = getattr(MAC, str(config['mac']))
            self.mac = mac_mod(self, config['MAC'])
            self.packet_length = self.mac.data_packet_length
        # except AttributeError:
        # raise ConfigError("Can't find MAC: %s" % config['mac'])
        except:
            raise

        ##############################
        # Routing
        ##############################
        # try:
        net_mod = getattr(Net, str(config['net']))
        self.net = net_mod(self, config['Network'])
        # except AttributeError as e:
        # raise ConfigError("Can't find Network: {}: {}".format(config['net'], e))

    def activate(self, rx_handler, monitor_mode=False):
        """
        Fired on Sim Start, activates the MAC layer and sets the packet length
        based on that activation.

        :param rx_handler: func: callback function to handle received packets
        :param monitor_mode: bool: enable notification of routed packets as well
        """
        self.app_rx_handler = rx_handler
        self.mac.activate()
        self.monitor_mode = monitor_mode
        self.packet_length = self.mac.data_packet_length

    def send(self, payload):
        """
        Initialise payload transmission down the stack
        """
        self.net.send_packet(payload)

    def recv(self, payload):
        """
        Trigger reception action from below
        """
        if self.app_rx_handler:
            self.app_rx_handler(payload)
        else:
            raise NotImplementedError("No App RX Handler configured")

    def signal_good_tx(self, packetid):
        for hdlr in self.tx_good_signal_hdlrs:
            hdlr(packetid)

    def signal_lost_tx(self, packetid):
        for hdlr in self.tx_lost_signal_hdlrs:
            hdlr(packetid)

    def get_current_position(self):
        """
        Host position as far as it's concerned (unless overridden with 'true' which will return the real physical location
        :return:
        """
        return self.host.get_pos()

    def get_real_current_position(self):
        """
        Host position as far as it's concerned (unless overridden with 'true' which will return the real physical location
        :return:
        """
        return self.host.get_pos(True)

    def query_drop_forward(self, packet):
        """
        Application Layer Call Back for Packet Forwarding
        :param packet:
        :return: bool
        """
        if self.fwd_signal_hdlrs:
            return (any([hdlr(packet) for hdlr in self.fwd_signal_hdlrs]))
        else:
            return False

    def query_pwr_adjust(self, packet):
        """
        Enables the application to adjust the power of a given outgoing packet
        :param packet:
        :return:
        """
        if self.pwd_signal_hdlr:
            return self.pwd_signal_hdlr(packet)
        else:
            return packet["level"]


