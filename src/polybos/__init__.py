#!/usr/bin/env python
"""
 * This file is part of the Aietes Framework
 *  (https://github.com/andrewbolster/aietes)
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
from __future__ import division

__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import os
import gc
import errno
import sys
import tempfile
from uuid import uuid4 as get_uuid
import logging
from copy import deepcopy
from datetime import datetime
from pprint import pformat
from natsort import natsorted
import pickle
import time
import collections

from configobj import ConfigObj
import validate
import numpy as np

# Must use the aietes path to get the config files
from aietes import Simulation
import aietes.Threaded
from aietes.Tools import _config_spec, _config_dir, _results_dir, nameGeneration, updateDict, kwarger, getConfig, ConfigError, try_x_times, secondsToStr, dotdict, notify_desktop, AutoSyncShelf, is_valid_aietes_datafile
from bounos import DataPackage, printAnalysis, load_sources, npz_in_dir

try:
    from contrib.Ghia.ecea.data_grapher import data_grapher

    ghia = True
except ImportError:
    ghia = False

# Mask in-sim progress display and let joblib do it's... job...
progress_display = False

PseudoScenario = collections.namedtuple("PseudoScenario",
                                        ["title", "datarun"]
)
logging.basicConfig()
log = logging.getLogger(__name__)


class Scenario(object):
    """ Scenario Object

    The Scenario Object deals with config management and passthrough, as well as some optional
    execution characteristics. The purpose of this manager is to abstract as much as humanly
    possible.

    """
    mutable_node_configs = {
        'behaviour': ['Behaviour', 'protocol'],
        'repulsion': ['Behaviour', 'repulsive_factor'],
        'schooling': ['Behaviour', 'schooling_factor'],
        'clumping': ['Behaviour', 'clumping_factor'],
        'waypointing': ['Behaviour', 'waypoint_factor'],
        'waypoint_style': ['Behaviour', 'waypoint_style'],
        'drifting': ['drift'],
        'ecea': ['ecea'],
        'positioning': ['position_generation'],
        'fudging': ['Behaviour', 'positional_accuracy'],
        'beacon_rate': ['beacon_rate'],
        'drift_dvl_scale': ['drift_scales', 'dvl'],
        'drift_gyro_scale': ['drift_scales', 'gyro'],
        'tof_type': ['tof_type'],
        'drift_noises': ['drift_noises'],
        'net': ['Network', 'protocol'],
        'mac': ['MAC', 'protocol'],
        'app': ['Application', 'protocol'],
        'app_rate': ['Application', 'packet_rate'],
        'app_length': ['Application', 'packet_length'],
    }

    def __init__(self, default_config=None, default_config_file=None,
                 runcount=1, title=None, *args, **kwargs):
        """
        Builds an initial config and divides it up for convenience later
            Can take default_config = <ConfigObj> or default_config_file = <path>
        Args:
            default_config: ConfigObj
            default_config_file(str): Path to config file
            runcount(int): Number of repeated executions of this scenario; this value can be
                overridden in the run method
            title(str)
        """

        if default_config_file is None and default_config is None:
            logging.info("Assume that the user wants a generic default")
            self._default_config = getConfig()
        elif isinstance(default_config, ConfigObj):
            logging.info("User provided a (hopefully complete) confobj")
            self._default_config = deepcopy(default_config)
        elif isinstance(default_config, dotdict):
            logging.info("User provided a (hopefully complete) dotdict")
            self._default_config = deepcopy(ConfigObj(default_config))
        elif default_config_file is not None:
            logging.info("User provided a config file that we have to interpolate " \
                         "against the defaults to generate a full config")
            intermediate_config = getConfig(default_config_file)
            self._default_config = Simulation.populateConfig(
                intermediate_config, retain_default=True
            )
        else:
            raise RuntimeError(
                "Given invalid Config of type %s: %s"
                % (type(self._default_config), self._default_config))

        if isinstance(self._default_config, ConfigObj):
            self._default_config_dict = self._default_config.dict()
        else:
            self._default_config_dict = ConfigObj(self._default_config)
        self._default_node_config = ConfigObj(
            self._default_config_dict['Node']['Nodes'].pop("__default__")
        )
        self.simulation = self._default_sim_config = ConfigObj(
            self._default_config_dict['Simulation']
        )
        self.environment = self._default_env_config = ConfigObj(
            self._default_config_dict['Environment']
        )
        self._default_run_count = runcount
        self.node_count = self._default_node_count = self._default_config_dict[
            'Node']['count']
        self.nodes = {}
        self.title = "DefaultScenario" if title is None else title

        self.committed = False

        self.tweaks = {}

        # Update Nodes List with Custom Nodes if any
        self._default_custom_nodes = ConfigObj(self._default_config_dict['Node']['Nodes'])
        for node_name, node_config in self._default_custom_nodes.items():
            # Casting to ConfigObj is a nasty hack for picklability (i.e. dotdict
            # subclasses dict but pickle protocol looks after dict natively.
            if node_name != "__default__":
                self.nodes[node_name] = deepcopy(ConfigObj(node_config))

        # May be unnecessary
        self._default_behaviour_dict = self.getBehaviourDict()


    def run(self, runcount=None, runtime=None, *args, **kwargs):
        """
        Offload this to AIETES
        Args:
            :param runcount(int): Number of repeated executions of this scenario; this value overrides the
                value set on init
            :param runtime(int): Override simulation duration (normally inherited from config)
        """
        if runcount is None:
            runcount = self._default_run_count

        self.mypath = os.path.join(
            kwargs.get("basepath", tempfile.mkdtemp()), self.title)
        os.makedirs(self.mypath)

        pp_defaults = {'outputFile': self.title, 'dataFile': kwargs.get(
            "dataFile", True), 'outputPath': self.mypath}
        self.datarun = [None for _ in range(runcount)]
        for run in range(runcount):
            if runcount > 1:
                pp_defaults.update({'outputFile': "%s-%d" % (self.title, run)})
            sys.stdout.write("%s," % pp_defaults['outputFile'])
            sys.stdout.flush()
            try:
                title = self.title + "-%s" % run
                sim = Simulation(config=self.config,
                                 title=title,
                                 logtofile=os.path.join(
                                     self.mypath, "%s.log" % title),
                                 logtoconsole=logging.ERROR,
                                 progress_display=progress_display
                )
                prep_stats = sim.prepare(sim_time=runtime)
                protected_run = try_x_times(10, RuntimeError, RuntimeError("Attempted ten runs, all failed"),
                                            sim.simulate)
                sim_time = protected_run()
                return_dict = sim.postProcess(**pp_defaults)
                data_retention_policy = kwargs.get("retain_data", False)
                # Implicitly implies boolean datatype
                if data_retention_policy is True:
                    self.datarun[run] = sim.generateDataPackage()
                elif data_retention_policy == "additional_only":
                    dp = sim.generateDataPackage()
                    self.datarun[run] = dp.additional.copy()
                elif data_retention_policy == "file":
                    self.datarun[run] = sim.generateDataPackage().write(title)
                else:
                    self.datarun[run] = return_dict
                log.info("%s(%s):%f%%"
                         % (run, return_dict['data_file'],
                            100.0 * float(sim_time) / prep_stats['sim_time']))

            except Exception:
                raise
        log.info("done %d runs for %d each" % (runcount, sim_time))

    def run_parallel(self, runcount=None, runtime=None, *args, **kwargs):
        """
        Offload this to AIETES multiprocessing queue

        Args:
            runcount(int): Number of repeated executions of this scenario; this value overrides the
                value set on init
            runtime(int): Override simulation duration (normally inherited from config)
        """
        self.mypath = os.path.join(
            kwargs.get("basepath", tempfile.mkdtemp()), self.title)
        try:
            os.mkdir(self.mypath)
        except OSError, e:
            if e.errno == errno.EEXIST and os.path.isdir(self.mypath):
                pass
            else:
                raise
        if runcount is None:
            runcount = self._default_run_count

        pp_defaults = {'outputFile': self.title, 'dataFile': kwargs.get(
            "dataFile", True), 'outputPath': self.mypath}
        self.datarun = [None for _ in range(runcount)]
        self.runlist = []
        for run in range(runcount):
            if runcount > 1:
                pp_defaults.update(
                    {'outputFile': "%s(%d:%d)" % (self.title, run, runcount)})
            try:
                title = self.title + "-%s" % run
                self.runlist.append(
                    (
                        kwarger(config=self.config,
                                title=title,
                                logtofile=os.path.join(
                                    self.mypath, "%s.log" % title),
                                logtoconsole=logging.ERROR,
                                progress_display=progress_display,
                                sim_time=runtime),
                        deepcopy(pp_defaults),
                        kwargs.get("retain_data")
                    )
                )
            except Exception:
                raise
        self.datarun = aietes.Threaded.parallel_sim(self.runlist)
        assert all(
            r is not None for r in self.datarun), "All dataruns should be completed by now"
        log.info("Got responses")

        print("done %d runs in parallel" % (runcount))

    def generateRunStats(self, sim_run_dataset=None):
        """
        Recieving a bounos.datapackage, generate relevant stats
        This is nasty and I can't remember why I did it this way
        Returns:
            A list of dict's given from DataPackage.package_statistics()
        """

        if sim_run_dataset is None:
            stats = []
            for i, d in enumerate(self.datarun):
                stats.append(self.generateRunStats(d))
            return stats
        elif isinstance(sim_run_dataset, DataPackage):
            return sim_run_dataset.package_statistics()
        else:
            raise RuntimeError("Cannot process simulation statistics of non-DataPackage: (%s)%s"
                               % (type(sim_run_dataset), sim_run_dataset))

    def write(self):
        """
        Dump the datafiles into a path but creating a folder with our name
        """
        for i, d in enumerate(self.datarun):
            filename = os.path.join(self.mypath, str(i))
            if hasattr(d, 'write'):
                d.write(filename=filename)
            pickle.dump(d, open(filename, 'wb'))

    def commit(self):
        """
        'Lock' the scenario, generating the final config, filling in any 'empty' config sections
        Raises:
            RuntimeError: on attempting to commit and already committed scenario
        """
        if self.committed:
            raise (RuntimeError("Attempted to commit twice (or more)"))
        print("Scenario {} Committed with {} nodes configured and {} defined".format(self.title,
                                                                                     len(self.nodes.keys(
                                                                                     )),
                                                                                     self.node_count))
        if self.node_count > len(self.nodes.keys()):
            self.addDefaultNode(count=self.node_count - len(self.nodes.keys()))

        self.config = self.generateConfig()
        self.committed = True

    def generateConfig(self):
        """
        Generate a config dict from the current state of the planned scenario
        Returns:
            DataPackage compatible dict
        """
        config = {}
        config['Simulation'] = self.simulation
        config['Environment'] = self.environment
        config['Node'] = {'Nodes': self.nodes,
                          'count': len([name for name in self.nodes.keys() if name != "__default__"])}
        return config

    def generateConfigObj(self):
        """
        Generate a ConfigObj from the current state of the planned scenario
        Returns:
            DataPackage compatible ConfigObj
        """
        rawconf = self.generateConfig()
        updateDict(
            rawconf, ['Node', 'Nodes', '__default__'], self._default_node_config)
        return ConfigObj(rawconf)

    def getBehaviourDict(self):
        """
        Generate and return a dict of currently configured behaviours wrt names of nodes
        eg. {'Waypoing':['alpha','beta','gamma'],'Flock':['omega']}

        Returns:
            dict of behaviours associated with a list of node names
        """
        default_bev = self._default_node_config['Behaviour']['protocol']

        if isinstance(default_bev, list) and len(default_bev > 1):
            behaviour_set = set(default_bev)
        else:
            behaviour_set = set()
            behaviour_set.add(default_bev)
        behaviours = {}
        behaviours[default_bev] = ['__default__', ]

        if self._default_node_config['bev'] != 'Null':
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
        """
        Set the scenario node count, but does not update the configuration (this is satisfied in
            the commit method)
        Args:
            count(int): New Node count
        """
        if self.committed:
            raise (
                RuntimeError("Attempted to modify scenario after committing"))
        if hasattr(self, "node_count"):
            print("Updating nodecount from %d to %d" %
                  (self.node_count, count))
        self.node_count = count

    def setDuration(self, tmax):
        """
        Set the scenario simulation duration,
        Args:
            tmax(int): New simulation time
        """
        if self.committed:
            raise (
                RuntimeError("Attempted to modify scenario after committing"))
        if hasattr(self.simulation, "sim_duration"):
            print("Updating simulation time from %d to %d"
                  % (self.simluation['sim_duration'], tmax))
        self.simulation['sim_duration'] = tmax

    def setEnvironment(self, environment):
        """
        Set the scenario simulation environment extent,
        Args:
            environment([int,int,int]): New simulation environment extent
        """
        if self.committed:
            raise (
                RuntimeError("Attempted to modify scenario after committing"))

        self.environment = environment

    def updateNode(self, node_conf, mutable, value):
        """
        Used to update selected field mappings between scenario definition and
            the scenario configspec, as defined in mutable_node_configs
        Args:
            node_conf(dict): current node configuration to be updated
            mutable(str): a string describing the aspect to be changed, present in the mutable map
            value(any): the mutable value to be set
        Raises:
            NotImplementedError: on invalid mutable key
        """
        if mutable is None:
            pass
        if mutable in self.mutable_node_configs:
            keys = self.mutable_node_configs[mutable]
            updateDict(node_conf, keys, value)
        else:
            raise NotImplementedError("Have no mutable map for %s" % mutable)

    def addCustomNode(self, variable_map, count=1):
        """
        Adds a node to the scenario based on a dict of mutable key,values
        Args:
            variable_map(dict): variables and values to be modified from the default
            count(int): if set, creates count instances of the custom node
        """
        node_conf = deepcopy(self._default_node_config)
        count = int(count)
        for variable, value in variable_map.iteritems():
            self.updateNode(node_conf, variable, value)
        self.addNode(node_conf=node_conf, count=count)

    def addDefaultNode(self, count=1):
        """
        Adds a default node
        Args:
            count(int): if set, creates count instances of the default node
        """
        node_conf = deepcopy(self._default_node_config)
        self.addNode(node_conf, count=count)

    def addNode(self, node_conf, names=None, count=1):
        """
        Adds a node to the scenario based on a (hopefully valid) node configuration
        Args:
            node_conf(dict): Fully defined node config dict
            names(list(str)): List of names for new nodes
            count(int): if set, creates count instances of the node
        Raises:
            RuntimeError if name definition doesn't make sense
        """
        if names is None:
            node_names = nameGeneration(
                count, existing_names=self.nodes.keys())
            if len(node_names) != count:
                raise RuntimeError("Names don't make any sense: Asked for %d, got %d: %s"
                                   % (count, len(node_names), node_names))
        elif isinstance(names, list):
            node_names = names
        else:
            raise RuntimeError("Names don't make any sense")

        for i, node_name in enumerate(node_names):
            self.nodes[node_name] = node_conf

    def updateDefaultNode(self, variable, value):
        """
        Update the default node for the scenario.
        Args:
            variable(str):The Variable to be modified (should me in the mutable map)
            value: the value to set that variable to
        Raises:
            RuntimeError if attempting to modify after commit.
        """
        if self.committed:
            raise RuntimeError(
                "Attempting to update default node config after committing")
        self.updateNode(self._default_node_config, variable, value)


class ExperimentManager(object):
    def __init__(self,
                 node_count=None,
                 title=None, parallel=False,
                 base_config_file=None,
                 base_exp_path=None, *args, **kwargs
    ):
        """
        The Experiment Manager Object deals with multiple scenarios build around a single or
            multiple experimental input. (Number of nodes, ratio of behaviours, etc)
        The purpose of this manager is to abstract the per scenario setup
        Args:
            :param node_count(int): Define the standard fleet size (Infer from Config)
            :param title(str): define a title for this experiment, to be used for file and folder naming,
                if not set, this defaults to a timecode and initialisation (not execution)
            :param base_config_file (str->FQPath): aietes config file to base the default scenario off of.

        """
        self._default_scenario = Scenario(title="__default__",
                                          default_config_file=base_config_file)
        self._base_config_file = base_config_file

        if title is None:
            self.title = "Default"
        else:
            self.title = title

        if not kwargs.get("no_time", False):
            self.title = "{}-{}".format(self.title, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

        self.exp_path = os.path.abspath(
            os.path.join(
                _results_dir if base_exp_path is None else base_exp_path,
                self.title
            )
        )
        try:
            if not os.path.isdir(self.exp_path):
                os.mkdir(self.exp_path)
        except:
            self.exp_path = tempfile.mkdtemp()
            print("Filepath collision, using %s" % self.exp_path)
        self.scenarios = {}

        if node_count is None:
            self._default_scenario.setNodeCount(self._default_scenario.node_count)
        else:
            self._default_scenario.setNodeCount(node_count)
        self.node_count = self._default_scenario.node_count
        self.parallel = parallel

    def updateDefaultNode(self, config_dict):
        """
        Applys a behaviour (given as a string) to the experimental default for node generation
        Args:
            behaviour(str): new default behaviour
        """
        for k, v in config_dict.iteritems():
            self._default_scenario.updateDefaultNode(k, v)

    def updateDefaultBehaviour(self, behaviour):
        """
        Applys a behaviour (given as a string) to the experimental default for node generation
        Args:
            behaviour(str): new default behaviour
        """
        self._default_scenario.updateDefaultNode('behaviour', behaviour)

    def run(self, runtime=None, runcount=None, retain_data=True, **kwargs):
        """
        Construct an execution environment and farm off simulation to scenarios
        Args:
            runtime(int): Override simulation duration (normally inherited from config)
            runcount(int): Number of repeated executions of this scenario; this value overrides the
                value set on init
            retain_data(bool/str): Decides wether the scenario state data is maintained or allowed to be GC'd
                    can be one of [True,'file','additional_only']
        """
        self.orig_path = os.path.abspath(_results_dir)
        self.runcount = runcount
        self.retain_data = retain_data
        kwargs.update(
            {"basepath": self.exp_path,
             "retain_data": self.retain_data,
             "runcount": self.runcount,
             "runtime": runtime})
        start = time.time()
        try:
            os.chdir(self.exp_path)
            # Q: Is this acting on the reference to scenario or the item in scenarios?
            for scenario_title, scenario in self.scenarios.items():
                scenario.commit()
                if self.parallel:
                    scenario.run_parallel(**kwargs)
                else:
                    scenario.run(**kwargs)
                self.scenarios[scenario_title] = scenario
                gc.collect()

        except ConfigError as e:
            print("Caught Configuration error %s on scenario config \n%s" %
                  (str(e), pformat(scenario.config)))
            raise
        finally:
            try:
                os.listdir(self.orig_path)
            except OSError as e:
                os.mkdir(self.orig_path)
            finally:
                os.chdir(self.orig_path)

            print("Experimental results stored in %s" % self.exp_path)
        self.runtime = time.time() - start
        msg = "Runtime:{}".format(secondsToStr(self.runtime))
        notify_desktop(msg)
        print(msg)

    def generateSimulationStats(self):
        """
        Returns:
            List of scenario stats (i.e. list of lists of run statistics dicts)
        """
        return {t: s.generateRunStats() for t, s in self.scenarios.items()}

    def updateNodeCounts(self, new_count):
        """
        Updates the node-count across all scenarios

        Args:
            new_count(int):new value to be used across scenarios
        """
        for t in self.scenarios.keys():
            self.scenarios[t].updateNodeCounts(new_count)

    def updateDuration(self, tmax):
        """
        Update the simulation time of currently configured scenarios
        Args:
            tmax(int): update experiment simulation duration for all scenarios
        """
        for t in self.scenarios.keys():
            self.scenarios[t].setDuration(tmax)

    def updateEnvironment(self, environment):
        """
        Update the environment extent of currently configured scenarios
        Args:
            environment([int,int,int]): update experiment simulation environment for all scenarios
        """
        if isinstance(environment, np.ndarray) and environment.shape == (3,):
            for t in self.scenarios.items():
                self.scenarios[t].setEnvironment(environment)
        else:
            raise ValueError(
                "Incorrect Environment Type given:{}{}".format(environment, type(environment)))

    def addVariableRangeScenario(self, variable, value_range, title_range=None):
        """
        Add a scenario with a range of configuration values to the experimental run

        Args:
            variable(str): mutable value description
            value_range(range or generator): values to be tested against.
        """
        if title_range is None:
            title_range = ["{}({})".format(variable, v) for v in value_range]
        for i, v in enumerate(value_range):
            s = Scenario(title=title_range[i],
                         default_config=self._default_scenario.generateConfigObj())
            s.addCustomNode({variable: v}, count=self.node_count)
            self.scenarios[s.title] = s

    def addApplicationVariableScenario(self, variable, value_range, title_range=None):
        """
        Add a scenario with a range of application/Node configuration values to the
        experimental run

        This *UPDATES* the default nodes rather than adding custom ones

        Args:
            variable(str): mutable value description
            value_range(range or generator): values to be tested against.
        """
        if title_range is None:
            title_range = ["{}({})".format(variable, v) for v in value_range]
        for i, v in enumerate(value_range):
            s = Scenario(title=title_range[i],
                         default_config=self._default_scenario.generateConfigObj())
            for node, node_config in s.nodes.items():
                s.updateNode(node_config, variable, v)
            self.scenarios[s.title] = s

    def addVariableNodeScenario(self, node_range):
        """
        Add a scenario with a range of configuration values to the experimental run

        Args:
            variable(str): mutable value description
            value_range(range or generator): values to be tested against.
        """
        for node_count in node_range:
            s = Scenario(title="{}({})".format("Nodes", node_count),
                         default_config=self._default_scenario.generateConfigObj())
            s.addDefaultNode(node_count)
            self.scenarios[s.title] = s

    def addMinorityNBehaviourSuite(self, behaviour_list, n_minority=1, title="Behaviour"):
        """
        Generate scenarios based on a list of 'attacking' behaviours, i.e. minority behaviours

        Args:
            behaviour_list(list): minority behaviours
            n_minority(int): number of minority attackers in each scenario (optional)
        """
        for v in behaviour_list:
            s = Scenario(title="%s(%s)" % (title, v),
                         default_config=self._default_scenario.generateConfigObj())
            s.addCustomNode({"behaviour": v}, count=n_minority)
            s.addDefaultNode(count=self.node_count - n_minority)
            self.scenarios[s.title] = s

    def addVariable2RangeScenarios(self, v_dict):
        """
        Add a 2dim range of scenarios based on a dictionary of value ranges.
        This generates a meshgrid and samples scenarios across the 2dim space

        Args:
            v_dict(dict):{'variable':'value_range', 'variable':'value_range'}
        """
        meshkeys = v_dict.keys()
        meshlist = []
        [meshlist.append(v_dict[key]) for key in meshkeys]
        meshgrid = np.asarray(np.meshgrid(*v_dict.values()))
        # NOTE meshgrid indexing is reversed compared to keyname
        # i.e. meshgrid[:,key[-1],key[-2],...,key[0]]
        # However, doing anything more than two is insane...
        scelist = [meshgrid[:, j, i]
                   for j in range(meshgrid.shape[1])
                   for i in range(meshgrid.shape[2])]

        for tup in scelist:
            d = dict(zip(meshkeys, tup))
            s = Scenario(title=str(["%s(%f)" % (variable, v) for variable, v in d.iteritems()]),
                         default_config=self._default_scenario.generateConfigObj())
            s.addCustomNode(d, count=self.node_count)
            self.scenarios[s.title] = s

    def addRatioScenarios(self, badbehaviour, goodbehaviour=None):
        """
        Add scenarios based on a ratio of behaviours of identical nodes

        If goodbehaviour is not specified, then the default node configuration *should* be used
            for the remaining nodes
        Args:
            badbehaviour(str):Aietes behaviour definition string (i.e. modulename)
            goodbehaviour(str):Aietes behaviour definition string (i.e. modulename) (optional)
        """
        for ratio in np.linspace(start=0.0, stop=1.00, num=self.node_count + 1):
            title = "%s(%.2f%%)" % (badbehaviour, float(ratio) * 100)
            print(title)
            s = Scenario(
                title=title, default_config=self._default_scenario.generateConfigObj())
            count = int(ratio * self.node_count)
            invcount = int((1.0 - ratio) * self.node_count)
            s.addCustomNode({"behaviour": badbehaviour}, count=count)

            if goodbehaviour is not None:
                s.addCustomNode({"behaviour": goodbehaviour}, count=invcount)
            else:
                s.addDefaultNode(count=invcount)
            self.scenarios[s.title] = s

    def addDefaultScenario(self, runcount=1, title=None):
        """
        Stick to the defaults
        """
        for i in range(runcount):
            s = Scenario(default_config=self._default_scenario.generateConfigObj(),
                         title=title if title is not None else "{}({})".format(self.title, i))
            self.scenarios[s.title] = s

    def addPositionScalingRange(self, scale_range, title=None, basis_node_name='n1'):
        """
        Using the base_config_file, generate a range of scaled positions for nodes that are
        manually set (i.e. operates only on the 'initial_position' value

        ONLY DEALS IN 2D AND ASSUMES ALL Z-VALUES ARE THE SAME
        :param range:
        :param title:
        :return:
        """
        base_config = getConfig(self._base_config_file)
        env_shape = np.asarray(base_config['Environment']['shape'])
        node_positions = {k: np.asarray(v['initial_position'], dtype=float)
                          for k, v in base_config['Node']['Nodes'].items()
                          if v.has_key('initial_position')  # This filters out any semi-defined nodes
        }
        node_centroids = {k: np.append((v[0:2] - env_shape[0:2] / 2), 0.0) for k, v in node_positions.items()}
        delta = np.asarray(node_positions[basis_node_name])
        for scale in scale_range:
            new_positions = {k: v * scale + delta for k, v in node_centroids.items()}

            if np.all(0 < new_positions.values() < env_shape):
                new_config = deepcopy(base_config)
                for k, v in new_positions.items():
                    new_config['Node']['Nodes'][k]['initial_position'] = list(v)  # ndarrays make literal_eval cry

                s = Scenario(default_config=Simulation.populateConfig(new_config, retain_default=True),
                             title="{}({:.2f})".format(self.title, scale)
                )
                self.scenarios[s.title] = s
            else:
                raise ConfigError("Scale {} is outside the defined environment: {}".format(scale, new_positions))

    @staticmethod
    def printStats(experiment, verbose=False):
        """
        Perform and print a range of summary experiment statistics including
            Fleet Distance (sum of velocities),
            Fleet Efficiency (Distance per time per node),
            Stdev(INDA) (Proxy for fleet positional variability)
            Stdev(INDD) (Proxy for fleet positional efficiency)
            Max Achievement Count,
            Percentage completion rate (how much of the fleet got the top count)
        """

        if isinstance(experiment, ExperimentManager):
            # Running as proper experiment Manager instance, no modification
            # required
            if hasattr(experiment, 'scenarios'):
                scenario_dict = experiment.scenarios
            elif hasattr(experiment, 'scenarios_file'):
                scenario_dict = AutoSyncShelf(experiment.scenarios_file)
        elif isinstance(experiment, list) \
                and all([isinstance(entry, Scenario) for entry in experiment]):
            # Have been given list of Scenarios entities in a single
            # 'scenario', treat as normalo
            scenario_dict = {s.title: s for s in experiment}
        elif isinstance(experiment, list) \
                and all([isinstance(entry, DataPackage) for entry in experiment]):
            # Have been given list of DataPackage entities in a single
            # 'scenario', treat as single virtual scenario
            scenario_dict = {dp.title: PseudoScenario(dp.title, dp) for dp in experiment}
        else:
            raise RuntimeWarning("Cannot validate experiment structure")

        def avg_of_dict(dict_list, keys):
            """
            Find the average of a key value across a list of dicts

            Args:
                dict_list(list of dict):list of value maps to be sampled
                keys(list of str): key-path of value in dict
            Returns:
                average value (float)
            """
            sum = 0
            count = 0
            for d in dict_list:
                count += 1
                for key in keys[:-1]:
                    d = d.get(key)
                sum += d[keys[-1]]
            return float(sum) / count

        correctness_stats = {}
        print(
            "Run\tFleet D, Efficiency\tstd(INDA,INDD)\tAch., Completion Rate\tCorrect/Confident\tSuspect ")
        for t, s in scenario_dict.items():
            correctness_stats[t] = []
            stats = [d.package_statistics() for d in s.datarun]
            # stats = temp_pool.map(lambda d: d.package_statistics(),s.datarun)
            suspects = []
            if isinstance(s, Scenario):
                # Running on a real scenario so use information we shouldn't
                # have
                suspect_behaviour_list = [(bev, nodelist)
                                          for bev, nodelist in s.getBehaviourDict().iteritems()
                                          if '__default__' not in nodelist]
                for _, nodelist in suspect_behaviour_list:
                    for node in nodelist:
                        suspects.append(node)
            print("%s,%s" % (t, suspects))

            for i, r in enumerate(stats):
                analysis = printAnalysis(s.datarun[i])
                confident = analysis['trust_stdev'] > 100
                correct_detection = (not bool(suspects) and not confident) or analysis[
                                                                                  'suspect_name'] in suspects
                correctness_stats[t].append(
                    (correct_detection, confident))
                if verbose:
                    print("%d\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%) %s, %s, %.2f, %.2f, %s" % (
                        i,
                        r['motion']['fleet_distance'], r[
                            'motion']['fleet_efficiency'],
                        r['motion']['std_of_INDA'], r['motion']['std_of_INDD'],
                        r['achievements']['max_ach'], r[
                            'achievements']['avg_completion'] * 100.0,
                        "%s(%.2f)" % (
                            str((correct_detection, confident)), analysis['trust_stdev']),
                        analysis['suspect_name'] + " %d" % analysis["suspect"],
                        analysis['suspect_distrust'],
                        analysis['suspect_confidence'],
                        str(analysis["trust_average"])
                    )
                    )

            print("AVG\t%.3fm (%.4f)\t%.2f, %.2f \t%d (%.0f%%)"
                  % (avg_of_dict(stats, ['motion', 'fleet_distance']),
                     avg_of_dict(stats, ['motion', 'fleet_efficiency']),
                     avg_of_dict(stats, ['motion', 'std_of_INDA']),
                     avg_of_dict(stats, ['motion', 'std_of_INDD']),
                     avg_of_dict(stats, ['achievements', 'max_ach']),
                     avg_of_dict(stats, ['achievements', 'avg_completion']) * 100.0))

        # Print out Correctness stats per scenario
        print("Scenario\t\t++\t+-\t-+\t--\t\t (Correct,Confident)")
        cct = cnt = nct = nnt = 0
        for run, stats in sorted(correctness_stats.items()):
            cc = sum([correct and confident for (correct, confident) in stats])
            cn = sum(
                [correct and not confident for (correct, confident) in stats])
            nc = sum(
                [not correct and confident for (correct, confident) in stats])
            nn = sum(
                [not correct and not confident for (correct, confident) in stats])
            print("%s\t%d\t%d\t%d\t%d" % (run, cc, cn, nc, nn))
            cct += cc
            cnt += cn
            nct += nc
            nnt += nn

        print("Subtot\t\t\t%d\t%d\t%d\t%d" % (cct, cnt, nct, nnt))
        print("Total\t\t\t%d\t\t\t%d" % (cct + cnt, nct + nnt))

        correct = cct + cnt
        confident = cct + nct
        incorrect = nct + nnt
        unconfident = cnt + nnt
        total = correct + incorrect
        n_scenarios = len(scenario_dict)

        print("Detection Accuracy pct with {} runs: {:.2%}".format(
            total, correct / total
        ))

        print("False-Confidence pct with {} runs: {:.2%}".format(
            total, nct / total
        ))

        print("True-Confidence pct with {} runs: {:.2%}".format(
            total, cct / total
        ))

        print("True-Confidence pct with {} runs factoring in the Waypoint state (i.e. positive negatives or missed-detections via lack of confidence): {:.2%}".format(
            total, (cnt - (total * (1 / n_scenarios))) / total
        ))

    def dump_dataruns(self):
        """
        Dump scenarios into the exp_path directory
        """
        for s in self.scenarios.values():
            s.write()

        if ghia:
            data_grapher(dir=self.exp_path, title=self.title)

    def dump_self(self):
        """
        Attempt to dump the entire experiment state
        """
        try:
            dumppath = os.path.abspath(
                os.path.join(self.exp_path, self.title + ".exp"))
            start = time.clock()
            print("Writing Experiment to {}".format(dumppath))
            # Scenarios have their own storage in self.scenarios_file
            self.scenarios.close()
            del self.scenarios
            pickle.dump(self, open(dumppath, 'wb'))
            print("Done in %f seconds" % (time.clock() - start))
        except Exception:
            import traceback

            print traceback.format_exc()
            print "Pickle Failed Miserably"

    def dump_analysis(self):
        """
        Ignore actual simulation information, record trust analysis stats to a pickle
        """
        s_paths = [None for _ in xrange(len(self.scenarios))]

        for t, s in self.scenarios.items():
            start = time.clock()
            s_path = os.path.abspath(
                os.path.join(self.exp_path, s.title + ".anl"))
            print("Writing analysis %s to %s" % (s.title, s_path))
            stats = [dict(
                printAnalysis(d).items() + d.package_statistics().items()) for d in s.datarun]

            pickle.dump(stats, open(s_path, "wb"))
            print("Done in %f seconds" % (time.clock() - start))

    @classmethod
    def write_analysis(exp):
        """
        big lazy multi-run write up that I'll change whenever I like

        CURRENTLY DOES
            Write intermediate fusion graphs for each scenario
            Best (Worst if Waypoint behaviour) case fusion presentation across N scenarios
        :param experiment:
        :return:
        """


class RecoveredExperiment(ExperimentManager):
    """
    SubClass to recover a partially executed experiment from an experiment directory.

    Not guaranteed to work in any way what so ever.
    :param dirpath:
    :return:
    """

    _shelf_name = "ScenarioDB.shelf"

    def __init__(self, dirpath):
        self.exp_path = os.path.abspath(dirpath)
        self.title = os.path.basename(dirpath)
        self.scenarios_file = os.path.join(dirpath, self._shelf_name)
        if os.path.isfile(self.scenarios_file):
            self.scenarios = AutoSyncShelf(self.scenarios_file)
        else:
            self.scenarios, self.scenarios_file = self.walk_dir(dirpath)

    @classmethod
    def walk_dir(cls, path):
        subdirs = filter(os.path.isdir,
                         map(lambda p: os.path.join(path, p),
                             os.listdir(path)
                         )
        )
        scenarios_file = os.path.join(path, cls._shelf_name)
        scenarios = AutoSyncShelf(scenarios_file)
        for subdir in natsorted(subdirs):
            title = os.path.basename(subdir)
            sources = npz_in_dir(subdir)
            scenarios[subdir] = PseudoScenario(subdir, load_sources(sources, comms_only=True))

        return scenarios, scenarios_file

