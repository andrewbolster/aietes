from aietes.Tools import Sim, ConfigError

import PHY, MAC, Net
import logging

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
            phy_mod = getattr(PHY,str(config['phy']))
        except AttributeError:
            raise ConfigError("Can't find PHY: %s"%config['phy'])

        self.phy = phy_mod(self,
                           self.channel_event,
                           self.config['PHY'])

        ##############################
        # MAC
        ##############################
        try:
            mac_mod=getattr(MAC,str(config['mac']))
        except AttributeError:
            raise ConfigError("Can't find MAC: %s"%config['mac'])
        self.mac = mac_mod(self,config['MAC'])


        ##############################
        # Routing
        ##############################
        try:
            net_mod=getattr(Net,str(config['net']))
        except AttributeError:
            raise ConfigError("Can't find Network: %s"%config['net'])

        self.net = net_mod(self,config['Network'])

    def activate(self):
        """
        Fired on Sim Start
        """
        self.mac.activate()

    def send(self,payload):
        """
        Initialise payload transmission down the stack
        """
        self.net.send(payload)

    def recv(self,payload):
        """
        Trigger reception action from below
        """
        self.app.recv()

