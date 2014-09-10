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

import logging

from aietes.Tools import ConfigError
import PHY
import MAC
import Net


class Layercake():
    """
    Defines the Four relevant network layers for a given node
    PHY,MAC,Network,Application
    """

    def __init__(self, host, config):
        """ Generate a Layercake with the stated config dict

        Args:
            host (Node):    This layercake's master node
            config (dict):  Layer-specific config settings

        Returns:
            A Layercake instance

        """

        self.host = host
        self.config = config
        self.channel_event = self.host.simulation.channel_event
        self.logger = host.logger
        self.sim_duration = host.simulation.duration_intervals
        ##############################
        # PHY
        ##############################
        try:
            phy_mod = getattr(PHY, str(config['phy']))
            self.phy = phy_mod(self,
                               self.channel_event,
                               self.config['PHY'])
        except KeyError:
            logging.warn("No PHY Configured")
        except AttributeError:
            raise ConfigError("Can't find PHY: %s" % config['phy'])


        ##############################
        # MAC
        ##############################
        try:
            mac_mod = getattr(MAC, str(config['mac']))
            self.mac = mac_mod(self, config['MAC'])
        except KeyError:
            logging.warn("No MAC Configured, activation is going to go wrong!")
        except AttributeError:
            raise ConfigError("Can't find MAC: %s" % config['mac'])


        ##############################
        # Routing
        ##############################
        try:
            net_mod = getattr(Net, str(config['net']))
            self.net = net_mod(self, config['Network'])
        except KeyError:
            logging.warn("No NET Configured")
        except AttributeError:
            raise ConfigError("Can't find Network: %s" % config['net'])


    def activate(self, rx_handler=None):
        """
        Fired on Sim Start
        """
        self.app_rx_handler = rx_handler
        self.mac.activate()

    def send(self, payload):
        """
        Initialise payload transmission down the stack
        """
        self.net.send(payload)

    def recv(self, payload):
        """
        Trigger reception action from below
        """
        if self.app_rx_handler:
            self.app_rx_handler(payload)
        else:
            raise NotImplementedError("No App RX Handler configured")

