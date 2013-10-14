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
 *     Andrew Bolster, Queen's University Belfast
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import sys
import os
import traceback
import optparse
import time
import logging
import cProfile
from pprint import pformat

import numpy as np


from datetime import datetime as dt
from Layercake import Layercake
from Environment import Environment
from Fleet import Fleet
from Node import Node
import Behaviour
from Animation import AIETESAnimation
from Tools import *
from bounos.DataPackage import DataPackage


np.set_printoptions(precision=3)


class Simulation():
    """
    Defines a single simulation
    Keyword Arguments:
        title:str(time)
        progress_display:bool(True)
        working_directory:str(/dev/shm)
        logtofile:str(None)
        logtoconsole:logging.level(INFO)
        logger:logging.logger(None)
        config_file:str(None)
        config:dict(None)
    """

    def __init__(self, *args, **kwargs):
        self._done = False
        self.title = kwargs.get("title", None )
        if self.title is None:
            self.title =  dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.progress_display = kwargs.get("progress_display", True)
        self.working_directory = kwargs.get("working_directory", "/dev/shm/")
        logtofile = kwargs.get("logtofile", None)
        logtoconsole = kwargs.get("logtoconsole", logging.INFO)

        if kwargs.get("logger", None) is None and __name__ in logging.Logger.manager.loggerDict:
            #Assume we need to make our own logger with NO preexisting handlers
            try:
                _tmplogdict = logging.Logger.manager.loggerDict[__name__]
                while len(_tmplogdict.handlers) > 0:
                    _tmplogdict.removeHandler(_tmplogdict.handlers[0])
            except KeyError:
                """Assumes that this is the first one"""
                pass
        self.logger = logging.getLogger(__name__)
        if logtoconsole is not None and not self.logger.root.handlers:
            #i.e. if logging to console and no handlers enabled
            ch = logging.StreamHandler()
            ch.setLevel(logtoconsole)
            ch.setFormatter(
                logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(ch)
            self.logger.info("Launched Console Logger (%s)"%log_level_lookup(ch.level))
        if logtofile is not None:
            hdlr = logging.FileHandler(logtofile)
            hdlr.setFormatter(logging.Formatter('[%(asctime)s] %(name)s-%(levelname)s-%(message)s'))
            hdlr.setLevel(logging.DEBUG)
            self.logger.addHandler(hdlr)
            self.logger.info("Launched File Logger (%s)"%logtofile)

        self.config_file = kwargs.get("config_file", None)
        self.config = kwargs.get("config", None)
        if self.config_file is None and self.config is None:
            self.logger.info("creating instance from default")
            self.config = validateConfig(None)
            self.config = self.generateConfig(self.config)
        elif self.config_file is None and self.config is not None:
            self.logger.info("using given config")
            config = validateConfig(self.config, final_check=True)
            config['Node']['Nodes'].pop('__default__')
            self.config = dotdict(config.dict())
        else:
            self.logger.info("creating instance from %s" % self.config_file)
            self.config = validateConfig(self.config_file)
            self.config = self.generateConfig(self.config)
        if "__default__" in self.config.Node.Nodes.keys():
            raise RuntimeError("Dun fucked up: __default__ node detected")

        #Once Validated we've got no reason to hold on to the configobj structure

        self.nodes = []
        self.fleets = []

    def prepare(self, waits=False, *args, **kwargs):
        """
        Args:
            waits(bool): set if running interactively (i.e. sim will wait for external actions)
            sim_time(int): override the simulation duration
        Raises:
            SystemExit on configuration error in setting log)
        Returns:
            Dict: {'sim_time':prepared sim duration (int),}
            This is an extensible interface that can be added to but must maintain compatibility.
        """
        # Attempt Validation and construct the simulation from that config.
        try:
            self.logger.setLevel(LOGLEVELS.get(self.config.get("log_level"), logging.NOTSET))
        except ConfigError as err:
            self.logger.error("Error in configuration, cannot continue: %s" % err)
            raise SystemExit(1)

        # Initialise simulation environment and configure a global channel event
        self.waits = waits
        Sim.initialize()
        self.channel_event = Sim.SimEvent(self.config.Simulation.channel_event_name)
        sim_time = kwargs.get('sim_time', None)
        if sim_time is not None:
            self.config.Simulation.sim_duration = int(sim_time)

        self.duration_intervals = self.config.Simulation.sim_duration / self.config.Simulation.sim_interval

        self.environment = self.configureEnvironment(self.config.Environment)
        self.nodes = self.configureNodes()

        # Single Fleet to control all
        self.fleets.append(Fleet(self.nodes, self))

        # Set up 'join-like' operation for nodes
        self.move_flag = Sim.Resource(capacity=len(self.nodes))
        self.process_flag = Sim.Resource(capacity=len(self.nodes))
        return {'sim_time': self.config.Simulation.sim_duration}

    def simulate(self, callback=None):
        """
        Initiate the processed Simulation
        Args:
            callback(func): Callback function to be called at each execution step
        Returns:
            Simulation Duration in ticks (generally seconds)
        """
        self.logger.info("Initialising Simulation %s, to run for %s steps" % (self.title, self.duration_intervals))
        for fleet in self.fleets:
            fleet.activate()
        if callback is not None:
            self.logger.info("Running with Callback: %s" % str(callback))
            Sim.startStepping()
        try:
            Sim.simulate(until=self.duration_intervals, callback=callback)
            self.logger.info("Finished Simulation at %s" % Sim.now())
        except RuntimeError as err:
            self.logger.exception("Simulation crashed at %s" % Sim.now())
            raise
        return Sim.now()

    def inner_join(self):
        """
        When all nodes have a move flag and none are processing
        """
        joined = self.move_flag.n == 0 and self.process_flag.n == len(self.nodes)
        if joined and debug:
            self.logger.debug("Joined: %s,%s" % (self.move_flag.n, self.process_flag.n))
        return joined

    def outer_join(self):
        """
        When all nodes have a processing flag and none are moving
        """
        joined = self.move_flag.n == len(self.nodes) and self.process_flag.n == 0
        if joined and debug:
            self.logger.debug("Joined: %s,%s" % (self.move_flag.n, self.process_flag.n))
        return joined

    def reverse_node_lookup(self, uuid):
        """Return Node Given UUID
        """
        for n in self.nodes:
            if n.id == uuid:
                return n
        raise KeyError("Given UUID does not exist in Nodes list")

    def now(self):
        return Sim.now()

    def currentState(self):
        positions = []
        vectors = []
        names = []
        contributions = []
        achievements = []
        for node in self.nodes:
            # Universal Stats
            pos = node.pos_log[:, :Sim.now()]
            vec = node.vec_log[:, :Sim.now()]
            pos = pos[np.isfinite(pos)].reshape(3, -1)
            vec = vec[np.isfinite(vec)].reshape(3, -1)

            positions.append(pos)
            vectors.append(vec)
            names.append(node.name)
            contributions.append(node.contributions_log)
            achievements.append(node.achievements_log)

        state = {'p': np.asarray(positions),
                'v': np.asarray(vectors),
                'names': names,
                'environment': self.environment.shape,
                'contributions': np.asarray(contributions),
                'achievements': np.asarray(achievements),
                'config': self.config,
                'title': self.title
        }

        # 'Quirky' Optional State Info

        #If any node is using waypoint bev, grab it.

        if any([isinstance(node.behaviour,Behaviour.WaypointMixin) for node in self.nodes]):
            waypointss = [getattr(node.behaviour, "waypoints", None) for node in self.nodes]
            waypointss = np.asarray(waypointss)
            #If All the valid waypoints are the same, only report one.
            if are_equal_waypoints(waypointss):
                state.update({'waypoints': waypointss[0]})
            else:
                state.update({'waypoints': waypointss})
        return state

    def generateConfig(self, config):
        #
        # NODE CONFIGURATION
        #
        preconfigured_nodes_count = 0
        pre_node_names = []
        nodes_config = {}
        node_default_config_dict = dotdict(config['Node']['Nodes'].pop('__default__').dict())
        config_dict = dotdict(config.dict())
        # Add the stuff we know whould be there...
        self.logger.debug("Initial Node Config from %s: %s" % (self.config_file, pformat(node_default_config_dict)))
        node_default_config_dict.update(
            # TODO import PHY,Behaviour, etc into the node config?
        )

        #
        # Check if there are individually configured nodes
        if isinstance(config_dict.Node.Nodes, dict) and len(config_dict.Node.Nodes) > 0:
            #
            # There Are Detailed Config Instances
            preconfigured_nodes_count = len(config_dict.Node.Nodes)
            self.logger.info("Have %d nodes from config: %s" % (
                preconfigured_nodes_count,
                nodes_config)
            )
            pre_node_names = config_dict.Node.Nodes.keys()

        #
        # Check and generate application distribution
        #   i.e. app = ["App A","App B"]
        #        dist = [ 4, 5 ]
        try:
            app = node_default_config_dict.Application.protocol
            dist = node_default_config_dict.Application.distribution
            nodes_count = config_dict.Node.count
        except AttributeError as e:
            self.logger.error("Error:%s" % e)
            self.logger.info("%s" % pformat(node_default_config_dict))
            raise ConfigError("Something is badly wrong")

        # Boundary checks:
        #   len(app)==len(dist)
        #   len(app) % nodes_count-preconfigured_nodes_count = 0
        if isinstance(app, list) and isinstance(dist, list):
            if len(app) == len(dist) and (nodes_count - preconfigured_nodes_count) % len(app) == 0:
                applications = [str(a)
                                for a, n in zip(app, dist)
                                for i in range(int(n))
                ]
                self.logger.debug("Distributed Applications:%s" % applications)
            else:
                raise ConfigError(
                    "Application / Distribution mismatch"
                )
        else:
            applications = [str(app) for i in range(int(nodes_count - preconfigured_nodes_count))]
            self.logger.info("Using Application:%s" % applications)

        #
        # Check and generate behaviour distribution
        #   i.e. bev = ["Bev A","Bev B"]
        #        dist = [ 4, 5 ]
        try:
            bev = node_default_config_dict.Behaviour.protocol
            dist = node_default_config_dict.Behaviour.distribution
            nodes_count = config_dict.Node.count
        except AttributeError as e:
            self.logger.error("Error:%s" % e)
            self.logger.info("%s" % pformat(node_default_config_dict))
            raise ConfigError("Something is badly wrong")

        # Boundary checks:
        #   len(bev)==len(dist)
        #   len(bev) % nodes_count-preconfigured_nodes_count = 0
        if isinstance(bev, list) and isinstance(dist, list):
            if len(bev) == len(dist) and (nodes_count - preconfigured_nodes_count) % len(bev) == 0:
                behaviours = [str(a)
                              for a, n in zip(bev, dist)
                              for i in range(int(n))
                ]
                self.logger.debug("Distributed behaviours:%s" % behaviours)
            else:
                raise ConfigError(
                    "Application / Distribution mismatch"
                )
        else:
            behaviours = [str(bev) for i in range(int(nodes_count - preconfigured_nodes_count))]
            self.logger.info("Using Behaviour:%s" % behaviours)
            #
        # Generate Names for any remaining auto-config nodes
        auto_node_names = nameGeneration(
            count=nodes_count - preconfigured_nodes_count,
            naming_convention=config_dict.Node.naming_convention
        )
        node_names = auto_node_names + pre_node_names

        # Give defaults to all
        for node_name in node_names:
            # Bare Dict/update instead of copy()
            nodes_config[node_name] = {}
            nodes_config[node_name].update(node_default_config_dict)

        # Give auto-config default
        for node_name, node_app in zip(auto_node_names, applications):
            # Add derived application
            nodes_config[node_name]['app'] = str(node_app)

        # Overlay Preconfigured with their own settings
        for node_name, node_config in config_dict.Node.Nodes.items():
            # Import the magic!
            nodes_config[node_name].update(node_config)

        #
        # Confirm
        #
        config_dict.Node.Nodes.update(nodes_config)
        self.logger.debug("Built Config: %s" % pformat(config))
        return config_dict

    def configureEnvironment(self, env_config):
        """
        Configure the physical environment within which the simulation executed
        Assumes empty unless told otherwise
        """
        return Environment(
            self,
            shape=env_config.shape,
            resolution=env_config.resolution,
            base_depth=env_config.base_depth
        )

    def configureNodes(self):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        node_list = []
        #
        # Configure specified nodes
        #
        for node_name, config in self.config.Node.Nodes.items():
            self.logger.debug("Generating node %s with config %s" % (node_name, config))
            new_node = Node(
                node_name,
                self,
                config,
                vector=self.vectorGen(node_name, config)
            )
            node_list.append(new_node)

        if len(node_list)>0:
            return node_list
        else:
            raise ConfigError("Node Generation failed: Zero nodes with config %s"%str(self.config))

    def vectorGen(self, node_name, node_config):
        """
        If a node is named in the configuration file, use the defined initial vector
        otherwise, use configured behaviour to assign an initial vector
        """
        try:  # If there is an entry, use it
            vector = node_config['initial_vector']
            self.logger.info("Gave node %s a configured initial vector: %s" % (node_name, str(vector)))
        except KeyError:
            gen_style = node_config['position_generation']
            if gen_style == "random":
                vector = self.environment.random_position()
                self.logger.debug("Gave node %s a random vector: %s" % (node_name, vector))
            elif gen_style == "center":
                vector = self.environment.position_around()
                self.logger.debug("Gave node %s a center vector: %s" % (node_name, vector))
            else:
                vector = [0, 0, 0]
                self.logger.debug("Gave node %s a zero vector from %s" % (node_name, gen_style))
        assert len(vector) == 3, "Incorrectly sized vector"

        return vector

    def generateDataPackage(self, *args, **kwargs):
        """
        Creates a bounos.DataPackage object from the current sim
        """
        from bounos.DataPackage import DataPackage

        dp = DataPackage(**(self.currentState()))
        return dp


    def postProcess(self, log=None, outputFile=None, displayFrames=None, dataFile=False, movieFile=False, gif=False,
                    inputFile=None, plot = False, xRes=1024, yRes=768, fps=24, extent=True):
        """
        Performs output and positions generation for a given simulation
        """

        def updatelines(i, positions, lines, displayFrames):
            """
            Update the currently displayed line positions
            positions contains [x,y,z],[t] positions for each vector
            displayFrames configures the display cache size
            """
            if isinstance(displayFrames, int):
                j = max(i - displayFrames, 0)
            else:
                j = 0
            for line, dat in zip(lines, positions):
                line.set_data(dat[0:2, j:i])  # x,y axis
                line.set_3d_properties(dat[2, j:i])  # z axis

            return lines

        dp = DataPackage(**(self.currentState()))



        positions = dp.p
        names = dp.names
        shape = dp.environment

        n_frames = len(positions[0][0])
        filename = "%s.aietes" % outputFile
        return_dict = {}

        if movieFile or plot or gif:
            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d.axes3d as axes3

            dpi = 80
            ipp = 80
            fig = plt.figure(dpi=dpi, figsize=(xRes / ipp, yRes / ipp))
            ax = axes3.Axes3D(fig)
            lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], label=names[i])[0] for i, dat in
                     enumerate(positions)]

            ax.legend()
            if extent is True:
                extent = tuple(zip(np.min(np.min(dp.p,axis=0),axis=1),np.max(np.max(dp.p,axis=0),axis=1)))
                (lx,rx),(ly,ry),(lz,rz) = extent
                x_width = abs(lx - rx)
                y_width = abs(ly - ry)
                z_width = abs(lz - rz)
                width = max(x_width, y_width, z_width) * 1.2

                avg = np.average(np.average(positions, axis=0), axis=1)
                ax.set_xlim3d((avg[0] - (width / 2), avg[0] + (width / 2)))
                ax.set_ylim3d((avg[1] - (width / 2), avg[1] + (width / 2)))
                ax.set_zlim3d((avg[2] - (width / 2), avg[2] + (width / 2)))
            else:
                ax.set_xlim3d((0, shape[0]))
                ax.set_ylim3d((0, shape[1]))
                ax.set_zlim3d((0, shape[2]))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            line_ani = AIETESAnimation(fig, updatelines, frames=int(n_frames), fargs=(positions, lines, displayFrames),
                                       interval=1000 / fps, repeat_delay=300, blit=True, )
            if movieFile:
                self.logger.info("Writing animation to %s.mp4" % filename)
                line_ani.save(
                          filename=filename,
                          fps=fps,
                          codec='mpeg4',
                          clear_temp=True
                )
                return_dict['ani_file'] = "%s.mp4" % filename
            if gif:
                self.logger.info("Writing animation to %s.gif" % filename)
                from matplotlib import animation as MPLanimation
                return_dict['ani_file'] = "%s.gif" % filename
                MPLanimation.FuncAnimation.save(line_ani, filename=return_dict['ani_file'], extra_args="-colors 8", writer='imagemagick', bitrate=-1, fps=fps)

        if outputFile is not None:
            if dataFile:
                ret_val = dp.write(filename)
                return_dict['data_file'] = ret_val[0]
                return_dict['config_file'] = ret_val[1]

        if plot:
            plt.show()
        return return_dict

    def deltaT(self, now, then):
        """
        Time in seconds between two simulation times
        """
        return (now - then) * self.config.Simulation.sim_interval


# Uncomment the following section if you want readline history support.
# import readline, atexit
# histfile = os.path.join(os.environ['HOME'], '.TODO_history')
# try:
#    readline.read_history_file(histfile)
# except IOError:
#    pass
# atexit.register(readline.write_history_file, histfile)


def go(options, args):
    logging.basicConfig(level=logging.INFO)
    sim = Simulation(config_file=options.config,
                     title=options.title,
                     logger=logging.getLogger('Aietes'),
                     logtoconsole=logging.ERROR if options.quiet else logging.INFO,
                     progress_display=not options.quiet
    )

    if options.input is None:
        sim.prepare(sim_time=options.sim_time)
        if not options.noexecution:
            try:
                sim.simulate()
            except RuntimeError as exp:
                print(exp)
                print("Will try to postprocess anyway")

    if options.movie or options.data or options.gif:
        print("Storing output in %s" % sim.title)
        sim.postProcess(inputFile=options.input, outputFile=sim.title, dataFile=options.data,
                        movieFile=options.movie, gif=options.gif, fps=options.fps)

    if options.plot:
        sim.postProcess(inputFile=options.input, displayFrames=720, plot=True, fps=options.fps)


def main():
    """
    Everyone knows what the main does; it does everything!
    """
    try:
        start_time = time.time()
        parser = optparse.OptionParser(
            formatter=optparse.TitledHelpFormatter(),
            usage=globals()['__doc__'],
            version='$Id: py.tpl 332 2008-10-21 22:24:52Z root $')
        parser.add_option('-q', '--quiet', action='store_true',
                          default=False, help='quiet output')
        parser.add_option('-v', '--verbose', action='store_true',
                          default=False, help='verbose output')
        parser.add_option('-P', '--profile', action='store_true',
                          default=False, help='profiled execution')
        parser.add_option('-p', '--plot', action='store_true',
                          default=False, help='perform plotting (overrides outputs)')
        parser.add_option('-m', '--movie', action='store_true',
                          default=None, help='generate and store movie (this takes a long time)')
        parser.add_option('-g', '--gif', action='store_true',
                          default=None, help='generate and store movie as an animated gif')
        parser.add_option('-f', '--fps', action='store', type="int",
                          default=24, help='set the fps for animation')
        parser.add_option('-x', '--noexecution', action='store_true',
                          default=False, help='prepare only, don\' execute simulation')
        parser.add_option('-d', '--data', action='store_true',
                          default=None, help='store output to datafile')
        parser.add_option('-i', '--input', action='store', dest='input',
                          default=None, help='store input file, this kills the simulation')
        parser.add_option('-t', '--tmax', action='store', dest='sim_time',
                          default=None, help='Override the simulation duration')
        parser.add_option('-c', '--config', action='store', dest='config',
                          default=None, help='generate simulation from config file')
        parser.add_option('-T', '--title', action='store', dest='title',
                          default=None,
                          help='Override the simulation name')
        parser.add_option('-r', '--runs', action='store', type="int",
                          default=1, help='set repeated runs (incompatible with Profiling)')
        (options, args) = parser.parse_args()
        print options
        exit_code = 0
        if options.verbose:
            print time.asctime()
        if options.title is None:                               # if no custom title
            if options.config is not None:                      # and have a config
                options.title = os.path.splitext(os.path.splitext(os.path.basename(options.config))[0])[
                    0]  # use config title
        if options.runs > 1:
            # During multiple runs, append _x to the run title
            basetitle = options.title
            for i in range(options.runs):
                options.title = "%s_%d" % (basetitle, i)
                exit_code = go(options, args)
        else:
            if options.profile:
                print "PROFILING"
                exit_code = cProfile.runctx("""go(options,args)""", globals(), locals(), filename="Aietes.profile")
            else:
                exit_code = go(options, args)
        if options.verbose:
            print time.asctime()
        if options.verbose:
            print 'TOTAL TIME IN MINUTES:',
        if options.verbose:
            print (time.time() - start_time) / 60.0
        sys.exit(exit_code)
    except KeyboardInterrupt, e:  # Ctrl-C
        raise e
    except SystemExit, e:  # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)


if __name__ == '__main__':
    main()
