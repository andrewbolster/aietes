import os
import sys
import tempfile
import logging
from copy import deepcopy

from configobj import ConfigObj
import validate
import numpy as np

from datetime import datetime
from aietes import _ROOT, Simulation # Must use the aietes path to get the config files
from aietes.Threaded import go as goSim
from aietes.Tools import nameGeneration, updateDict
from bounos import DataPackage


_config_spec = '%s/configs/default.conf' % _ROOT

mutable_node_configs = {
    'behaviour': ['Behaviour', 'protocol'],
    'repulsion': ['Behaviour', 'repulsive_factor'],
    'schooling': ['Behaviour', 'schooling_factor'],
    'clumping': ['Behaviour', 'clumping_factor'],
    'waypointing': ['Behaviour', 'waypoint_factor'],
    'fudging': ['Behaviour', 'positional_accuracy'],
}


def getConfig(source_config_file=None, config_spec=_config_spec):
    config = ConfigObj(source_config_file, configspec=config_spec, stringify=True, interpolation=True)
    config_status = config.validate(validate.Validator(), copy=True)
    if not config_status:
        if source_config_file is None:
            raise RuntimeError("Configspec is Broken: %s" % config_spec)
        else:
            raise RuntimeError("Configspec doesn't match given input structure: %s" % source_config_file)
    return config


class Scenario(object):
    """
    The Generic Manager Object deals with config management and passthrough, as well as some optional execution characteristics
        The purpose of this manager is to abstract as much as humanly possible
    Takes:
        default_config_file:str
        runcount:int
        title:str

    """

    def __init__(self, *args, **kwargs):
        """
        Builds an initial config and divides it up for convenience later
            Can take default_config = <ConfigObj> or default_config_file = <path>
        """

        self._default_config = kwargs.get("default_config", getConfig(kwargs.get("default_config_file", None)))
        if not isinstance(self._default_config, ConfigObj):
            raise RuntimeError(
                "Given invalid Config of type %s: %s" % (type(self._default_config), self._default_config))
        self._default_config_dict = self._default_config.dict()
        self._default_node_config = self._default_config_dict['Node']['Nodes'].pop("__default__")
        self._default_custom_nodes = self._default_config_dict['Node']['Nodes']
        self.simulation = self._default_sim_config = self._default_config_dict['Simulation']
        self.environment = self._default_env_config = self._default_config_dict['Environment']
        self._default_run_count = kwargs.get("runcount", 1)
        self.node_count = self._default_node_count = self._default_config_dict['Node']['count']
        self.nodes = {}
        self._default_behaviour_dict = self.getBehaviourDict()

        self.title = kwargs.get("title", "DefaultScenario")

        self.committed = False

        self.tweaks = {}

    def run(self, runcount=None, runtime=None, *args, **kwargs):
        """
        Offload this to AIETES
        Keyword Arguments:
            runcount:int(default)
            runtime:int(None)
        """
        if runcount is None:
            runcount = self._default_run_count

        pp_defaults = {'outputFile': self.title, 'dataFile': True}
        if not self.committed: self.commit()
        self.datarun = [None for _ in range(runcount)]
        for run in range(runcount):
            if runcount > 1:
                pp_defaults.update({'outputFile': "%s(%d:%d)" % (self.title, run, runcount)})
            sys.stdout.write("%s," % pp_defaults['outputFile'])
            sys.stdout.flush()
            try:
                sim = Simulation(config=self.config,
                                 title=self.title + "-%s" % run,
                                 logtofile=self.title + ".log",
                                 logtoconsole=logging.ERROR,
                                 progress_display=False
                )
                prep_stats = sim.prepare(sim_time=runtime)
                sim_stats = sim.simulate()
                return_dict = sim.postProcess(**pp_defaults)
                self.datarun[run] = sim.generateDataPackage()
                print return_dict['data_file']

            except Exception as exp:
                raise
        print("done %d runs for %d each" % (runcount, runtime if runtime is not None else -sim_stats))

    def runThreaded(self, *args, **kwargs):
        """
        Offload this to AIETES threaded
        Still borked...
        """
        runcount = kwargs.get("runcount", self._default_run_count)
        if not self.committed: self.commit()

        pp_defaults = {'outputFile': self.title + kwargs.get("title", ""), 'dataFile': True}
        sim_args = {'config': self.config, 'title': self.title, 'logtofile': self.title + ".log",
                    'logtoconsole': logging.INFO, 'progress_display': False}

        self.datarun = goSim(runcount, sim_args=sim_args, pp_args=pp_defaults)

    def generateRunStats(self, sim_run_dataset=None):
        """
        Recieving a bounos.datapackage, generate relevant stats
        """

        if sim_run_dataset is None:
            stats = []
            for i, d in enumerate(self.datarun):
                stats.append(self.generateRunStats(d))
            return stats
        elif isinstance(sim_run_dataset, DataPackage):
            return sim_run_dataset.package_statistics()
        else:
            raise RuntimeError("Cannot process simulation statistics of non-DataPackage: (%s)%s" % (
                type(sim_run_dataset), sim_run_dataset))

    def commit(self):
        print("Scenario Committed with %d nodes configured and %d defined" % (len(self.nodes.keys()), self.node_count))
        if self.node_count > len(self.nodes.keys()):
            self.addDefaultNode(count=self.node_count - len(self.nodes.keys()))

        self.config = self.generateConfig()
        self.committed = True

    def generateConfig(self):
        config = {}
        config['Simulation'] = self.simulation
        config['Environment'] = self.environment
        config['Node'] = {'Nodes': self.nodes,
                          'count': len(self.nodes.keys())}
        return config

    def generateConfigObj(self):
        rawconf = self.generateConfig()
        updateDict(rawconf, ['Node', 'Nodes', '__default__'], self._default_node_config)
        return ConfigObj(rawconf)

    def getBehaviourDict(self):
        default_bev = self._default_node_config['Behaviour']['protocol']

        behaviour_set = set(default_bev)
        behaviours = {}
        behaviours[default_bev] = ['__default__']

        if self._default_node_config['bev'] != 'Null': #Stupid string comparison stops this from being 'is not'
            raise NotImplementedError(
                "TODO Deal with parametric behaviour definition:%s" %
                self._default_node_config['bev'])
        if self._default_custom_nodes:
            for name, node in self._default_custom_nodes.iteritems():
                n_bev = node['Behaviour']['protocol']
                behaviour_set.add(n_bev)
                if n_bev in behaviours:
                    behaviours[n_bev].append(name)
                else:
                    behaviours[n_bev] = [name]
        if self.nodes:
            for name, node in self.nodes.iteritems():
                n_bev = node['Behaviour']['protocol']
                behaviour_set.add(n_bev)
                if n_bev in behaviours:
                    behaviours[n_bev].append(name)
                else:
                    behaviours[n_bev] = [name]
        return behaviours

    def setNodeCount(self, count):
        if hasattr(self, "node_count"):
            print("Updating nodecount from %d to %d" % (self.node_count, count))
        self.node_count = count

    def setDuration(self, tmax):
        self.simulation['sim_duration'] = tmax

    def updateNode(self, node_conf, mutable, value):
        if mutable is None:
            pass
        if mutable in mutable_node_configs:
            keys = mutable_node_configs[mutable]
            updateDict(node_conf, keys, value)
        else:
            raise NotImplementedError("Have no mutable map for %s" % mutable)

    def addCustomNode(self, variable_map, count=1):
        node_conf = deepcopy(self._default_node_config)
        count = int(count)
        for variable, value in variable_map.iteritems():
            self.updateNode(node_conf, variable, value)
        self.addNode(node_conf=node_conf, count=count)

    def addDefaultNode(self, count=1):
        node_conf = deepcopy(self._default_node_config)
        self.addNode(node_conf, count=count)

    def addNode(self, node_conf, names=None, count=1):
        if names is None:
            node_names = nameGeneration(count, existing_names=self.nodes.keys())
            if len(node_names) != count:
                raise RuntimeError("Names don't make any sense: Asked for %d, got %d: %s" % (
                    count,
                    len(node_names),
                    node_names)
                )
        elif isinstance(names, list):
            node_names = names
        else:
            raise RuntimeError("Names don't make any sense")

        for i, node_name in enumerate(node_names):
            self.nodes[node_name] = node_conf

    def updateDefaultNode(self, variable, value):
        self.updateNode(self._default_node_config, variable, value)


class ExperimentManager(object):
    """
    The Experiment Manager Object deals with multiple scenarios build around a single or multiple experimental input. (Number of nodes, ratio of behaviours, etc)
        The purpose of this manager is to abstract the per scenario setup
    Takes:
        node_count:int
        title:str

    """

    def __init__(self, node_count=4, title=None, *args, **kwargs):
        """
        Acquire Generic Scenario
        """
        self.scenarios = []
        self._default_scenario = Scenario(title="__default__")
        self.node_count = node_count
        if title is None:
            self.title = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.title = title
        self._default_scenario.setNodeCount(self.node_count)

    def updateBaseBehaviour(self, behaviour):
        self._default_scenario.updateDefaultNode('behaviour', behaviour)

    def run(self, threaded=False, **kwargs):
        """
        Construct an execution environment and farm off simulation to scenarios
        Takes:
            title:int
            runcount:int
            runtime:int(None)
        """
        title = kwargs.get("title", self.title)
        self.exp_path = os.path.abspath(os.path.join(os.path.curdir, title))
        self.orig_path = os.path.abspath(os.path.curdir)
        self.runcount = kwargs.get("runcount", 1)
        try:
            os.mkdir(self.exp_path)
        except:
            self.exp_path = tempfile.mkdtemp()
        try:
            os.chdir(self.exp_path)
            for scenario in self.scenarios:
                if threaded:
                    scenario.runThreaded(**kwargs)
                else:
                    scenario.run(**kwargs)
        finally:
            os.chdir(self.orig_path)
            print("Experimental results stored in %s" % self.exp_path)

    def generateSimulationStats(self):
        """
        Walks the scenario chain and returns a stats list of lists
        """
        return [s.generateRunStats() for s in self.scenarios]

    def updateNodeCounts(self, new_count):
        """
        Updates the node-count makeup; different behaviour ie new_count is list or scalar
        """
        if isinstance(new_count, list) and len(new_count) == len(self.scenarios):
            for i, s in enumerate(self.scenarios):
                s.updateNodeCounts(new_count[i])
        else:
            for s in self.scenarios:
                s.updateNodeCounts(new_count)

    def updateDuration(self, tmax):
        """
        Update the simulation time of currently configured scenarios
        """
        for s in self.scenarios:
            s.setDuration(tmax)

    def addVariableRangeScenario(self, variable, value_range):
        """
        Add a scenario with a range of configuration values to the experimental run
        """
        for v in value_range:
            s = Scenario(title="%s(%f)" % (variable, v), default_config=self._default_scenario.generateConfigObj())
            s.addCustomNode({variable: v}, count=self.node_count)
            self.scenarios.append(s)

    def addVariableAttackerBehaviourSuite(self, behaviour_list, n_attackers=1):
        """
        Add a scenario with a range of configuration values to the experimental run
        """
        for v in behaviour_list:
            s = Scenario(title="Behaviour(%s)" % (v), default_config=self._default_scenario.generateConfigObj())
            s.addCustomNode({"behaviour": v}, count=n_attackers)
            s.addDefaultNode(count=self.node_count - n_attackers)
            self.scenarios.append(s)

    def addVariable2Scenario(self, v_dict):
        """
        Add a 2dim range of scenarios based on a dictionary of {'variable':'value_range', 'variable':'value_range'}
        """
        meshkeys = v_dict.keys()
        meshlist = []
        [meshlist.append(v_dict[key]) for key in meshkeys]
        meshgrid = np.asarray(np.meshgrid(*v_dict.values()))
        # NOTE meshgrid indexing is reversed compared to keyname
        # i.e. meshgrid[:,key[-1],key[-2],...,key[0]]
        # However, doing anything more than two is insane...
        scelist = [meshgrid[:, j, i] for j in range(grid.shape[1]) for i in range(meshgrid.shape[2])]

        for tup in scelist:
            s = Scenario(title="%s(%f)" % (variable, v), default_config=self._default_scenario.generateConfigObj())
            s.addCustomNode({variable: v}, count=self.node_count)
            self.scenarios.append(s)


    def addRatioScenario(self, badbehaviour, goodbehaviour=None):
        """
        Add a scenario based on a ratio of behaviours of identical nodes
        If goodbehaviour is not specified, then the default node configuration *should* be used for the remaining
            nodes
        """
        for ratio in np.linspace(start=0.0, stop=1.00, num=self.node_count + 1):
            title = "%s(%.2f%%)" % (badbehaviour, float(ratio) * 100)
            print(title)
            s = Scenario(title=title, default_config=self._default_scenario.generateConfigObj())
            count = int(ratio * self.node_count)
            invcount = int((1.0 - ratio) * self.node_count)
            s.addCustomNode({"behaviour": badbehaviour}, count=count)

            if goodbehaviour is not None:
                s.addCustomNode({"behaviour": goodbehaviour}, count=invcount)
            else:
                s.addDefaultNode(count=invcount)
            self.scenarios.append(s)

    @staticmethod
    def printStats(exp):

        def avg_of_dict(dict_list, keys):
            sum = 0
            count = 0
            for d in dict_list:
                count += 1
                for key in keys[:-1]:
                    d = d.get(key)
                sum += d[keys[-1]]
            return float(sum) / count

        print("Run\tFleet D, Efficiency\tstd(INDA,INDD)\tAch., Completion Rate\t")
        for s in exp.scenarios:
            stats = s.generateRunStats()
            print("%s,%s" % (s.title, [(bev, nodelist)
                                       for bev, nodelist in s.getBehaviourDict().iteritems()
                                       if '__default__' not in nodelist
            ]))
            print("AVG\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)" % (
                avg_of_dict(stats, ['motion', 'fleet_distance']),
                avg_of_dict(stats, ['motion', 'fleet_efficiency']),
                avg_of_dict(stats, ['motion', 'std_of_INDA']),
                avg_of_dict(stats, ['motion', 'std_of_INDD']),
                avg_of_dict(stats, ['achievements', 'max_ach']),
                avg_of_dict(stats, ['achievements', 'avg_completion']) * 100.0
            )
            )

            for i, r in enumerate(stats):
                print("%d\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)" % (
                    i,
                    r['motion']['fleet_distance'], r['motion']['fleet_efficiency'],
                    r['motion']['std_of_INDA'], r['motion']['std_of_INDD'],
                    r['achievements']['max_ach'], r['achievements']['avg_completion'] * 100.0
                )
                )