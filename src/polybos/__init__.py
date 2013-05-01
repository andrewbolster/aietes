import os
import tempfile
from pprint import pprint

from configobj import ConfigObj
import validate
from datetime import datetime

import logging

from aietes import _ROOT, Simulation # Must use the aietes path to get the config files
from aietes.Tools import nameGeneration

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

class Scenario(object):
    """
    The Generic Manager Object deals with config management and passthrough, as well as some optional execution characteristics
        The purpose of this manager is to abstract as much as humanly possible
    """
    mutable_configs={
      'behaviour':['Behaviour','protocol'],
      'repulsion':['Behaviour','repulsive_factor'],
      'schooling':['Behaviour','schooling_factor'],
      'clumping':['Behaviour','clumping_factor'],
      'fudging':['Behaviour','positional_accuracy']
    }

    def __init__(self, *args, **kwargs):
        """
        Builds an initial config and divides it up for convenience later
        """
        self._default_config = getConfig()
        self._default_config_dict = self._default_config.dict()
        self._default_node_config = self._default_config_dict['Node']['Nodes'].pop("__default__")
        self._default_custom_nodes = self._default_config_dict['Node']['Nodes']
        self.simulation = self._default_sim_config = self._default_config_dict['Simulation']
        self.environment = self._default_env_config = self._default_config_dict['Environment']
        self._default_run_count = kwargs.get("runcount",1)
        self.node_count = self._default_node_count = self._default_config_dict['Node']['count']
        self._default_behaviour_dict = self.get_behaviour_dict()
        self.nodes={}
        self.title = kwargs.get("title",None)

        self.committed = False

        self.tweaks = {}

    def run(self, *args, **kwargs):
        """
        Offload this to AIETES
        """
        runcount = kwargs.get("runcount",self._default_run_count )
        pp_defaults={'outputFile':self.title+kwargs.get("title",""), 'dataFile':True}
        if not self.committed: self.commit()
        self.datarun=[None for _ in range(runcount)]
        for run in range(runcount):
            try:
                sim = Simulation(config = self.config, 
                                 title = self.title+"-%s"%run, 
                                 logtofile=self.title+".log", 
                                 logtoconsole=logging.INFO,
                                 progress_display=False
                                )
                prep_stats=sim.prepare()
                sim_stats=sim.simulate()
                sim.postProcess(**pp_defaults)
                self.datarun[run] = sim.generateDataPackage()
            except Exception as exp:
                raise

    def generate_simulation_stats(self, sim_run_dataset):
        """
        Recieving a bounos.datapackage, generate relevant stats
        """
        pass


    def commit(self):
        if self.node_count > len(self.nodes.keys()):
            self.add_default_node(count = self.node_count - len(self.nodes.keys()))

        self.config = self.generate_config()
        self.committed = True


    def generate_config(self):
        config = {}
        config['Simulation'] = self.simulation
        config['Environment'] = self.environment
        config['Node'] = {'Nodes':self.nodes, 'count':len(self.nodes.keys())}
        return config

    def get_behaviour_dict(self):
        default_bev = self._default_node_config['Behaviour']['protocol']

        behaviour_set = set(default_bev)
        behaviours = {}
        behaviours[default_bev]=['__default__']

        if self._default_node_config['bev'] != 'Null': #Stupid string comparison stops this from being 'is not'
            raise NotImplementedError("TODO Deal with parametric behaviour definition:%s"%self._default_node_config['bev'])
        if self._default_custom_nodes:
            for name,node in self._default_custom_nodes.iteritems():
                n_bev = node['Behaviour']['protocol']
                behaviour_set.add(n_bev)
                if n_bev in behaviours:
                  behaviours[n_bev].append(name)
        return behaviours

    def set_node_count(self, count):
        if hasattr("node_count",self):
            print("Updating nodecount from %d to %d"%(self.node_count, count))
        self.node_count = count

    def update_node(self, node_conf, mutable, value):
        if mutable is None:
            pass
        if mutable in self.mutable_configs:
            keys = self.mutable_configs[mutable]
            for key in keys[:-1]:
              node_conf = node_conf.setdefault(key,{})
            node_conf[keys[-1]]=value
        else:
            raise NotImplementedError("Have no mutable map for %s"%mutable)

    def add_custom_node(self,variable_map,count=1):
        node_conf = self._default_node_config.copy()
        for variable,value in variable_map.iteritems():
            self.update_node(node_conf,variable,value)
        node_names = nameGeneration(count,existing_names=self.nodes.keys())
        for node_name in node_names:
            self.nodes[node_name]=node_conf

    def add_default_node(self,count=1):
        node_conf = self._default_node_config.copy()
        node_names = nameGeneration(count,existing_names=self.nodes.keys())
        for node_name in node_names:
            self.nodes[node_name]=node_conf

    def update_default_node(self,variable,value):
        self.update_node(self._default_node_config, variable, value)



class ExperimentManager(object):
    """
    The Experiment Manager Object deals with multiple scenarios build around a single or multiple experimental input. (Number of nodes, ratio of behaviours, etc)
        The purpose of this manager is to abstract the per scenario setup
    """
    def __init__(self, *args, **kwargs):
        """
        Acquire Generic Scenario
        """
        self.scenarios = []
        self.node_count = None
        self.title = kwargs.get("title",datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    def run(self, *args, **kwargs):
        """
        Construct an execution environment and farm off simulation to scenarios
        """
        title = kwargs.get("title", self.title)
        self.exp_path = os.path.abspath(os.path.join(os.path.curdir,title))
        self.orig_path = os.path.abspath(os.path.curdir)
        self.runcount = kwargs.get("runcount",1)
        try:
            os.mkdir(self.exp_path)
        except:
            self.exp_path = tempfile.mkdtemp()
        try:
            os.chdir(self.exp_path)
            for scenario in self.scenarios:
                scenario.run(runcount = self.runcount)
        finally:
            os.chdir(self.orig_path)
            print("Experimental results stored in %s"%self.exp_path)

    def updateNodeCounts(self, new_count):
        """
        Updates the node-count makeup; different behaviour ir new_count is list or scalar
        """
        if  isinstance(new_count, list) and len(new_count) == len(scenarios):
            for i,s in enumerate(self.scenarios):
                s.updateNodeCounts(new_count[i])

    def addScenario(self, variable, value_range):
        """
        Add a scenario configuration to the experimental run
        """
        for v in value_range:
          s = Scenario(title="%s(%f)"%(variable,v))
          s.add_custom_node({variable:v})
          self.scenarios.append(s)


