from aietes.Tools import Sim, ConfigError
from aietes import Applications

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
        self.logger = host.logger

        ##############################
        # PHY
        ##############################
        try:
            phy_mod = getattr(PHY,str(config.phy))
        except AttributeError:
            raise ConfigError("Can't find PHY: %s"%config.phy)

        self.phy = self.host.config.phy_mod(self,config.channel_event,config.phy)

        ##############################
        # MAC
        ##############################
        try:
            mac_mod=getattr(MAC,str(config.mac))
        except AttributeError:
            raise ConfigError("Can't find MAC: %s"%config.mac)
        self.mac = mac_mod(self,config.mac)


        ##############################
        # Routing
        ##############################
        try:
            net_mod=getattr(Net,str(config.net))
        except AttributeError:
            raise ConfigError("Can't find Network: %s"%config.net)

        self.net = net_mod(self,config.net)

        ##############################
        # Application
        ##############################
        try:
            app_mod=getattr(Applications,str(config.app))
        except AttributeError:
            raise ConfigError("Can't find Application: %s"%config.app)
        self.app = app_mod(self,config.app)

    def activate(self):
        """
        Fired on Sim Start
        """
        self.mac.activate()
        self.app.activate()


