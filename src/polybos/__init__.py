# Polybos - Campaign generator for Aietes
from os import path
from pprint import pprint

from configobj import ConfigObj
import validate

from aietes import _ROOT

_config_spec = '%s/configs/default.conf'%_ROOT

def getConfig(source_config_file=None, config_spec = _config_spec):
    config = ConfigObj(source_config_file, configspec = config_spec, stringify = True, interpolation = True)
    config_status = config.validate(validate.Validator(), copy = True)
    if not config_status:
        if source_config_file is None:
            raise RuntimeError("Configspec is Broken: %s"%config_spec)
        else:
            raise RuntimeError("Configspec doesn't match given input structure: %s"%source_config_file)
    return config

class ConfigManager(object):

    """
    The ConfigManager Object deals with multiple configuration generation from a single (usually the default) Aietes config
    """
    def __init__(self, *args, **kwargs):

        """
        Builds an initial config and divides it up for convenience later
        """
        self._default_config = getConfig()
        self._default_config_dict = self._default_config.dict()
        self._default_node_config = self._default_config_dict['Node']['Nodes'].pop("__default__")
        self._default_custom_nodes = self._default_config_dict['Node']['Nodes']
        self._default_sim_config = self._default_config_dict['Simulation']
        self._default_env_config = self._default_config_dict['Environment']


