import sys
import os
import traceback
import optparse
import time
import logging
import cProfile
from datetime import datetime as dt
from pprint import pformat

import numpy as np
from configobj import ConfigObj
import validate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3

from Layercake import Layercake
from Environment import Environment
from Fleet import Fleet
from Node import Node
import Behaviour
from Animation import AIETESAnimation
from Tools import *

np.set_printoptions(precision=3)

_ROOT = os.path.abspath(os.path.dirname(__file__))


class Simulation():

    """
    Defines a single simulation
    """

    def __init__(self, *args, **kwargs):
        self.config_spec = '%s/configs/default.conf' % _ROOT
        self.title = kwargs.get("title", None)
        logtofile = kwargs.get("logtofile", None)
        if logtofile is not None:
            setlogtofile(logtofile)
        self.logger = baselogger.getChild("%s" % self.__class__.__name__)
        self.config_file = kwargs.get("config_file", None)
        self.config = kwargs.get("config", None)
        if self.config_file is None and self.config is None:
            self.logger.info("creating instance from default")
            self.config = self.validateConfig(None)
            self.config = self.generateConfig(self.config)
        elif self.config_file is None and self.config is not None:
            self.logger.info("using given config")
            config = self.validateConfig(self.config, final_check = True)
            config['Node']['Nodes'].pop('__default__')
            self.config = dotdict(config.dict())
        else:
            self.logger.info("creating instance from %s" % self.config_file)
            self.config = self.validateConfig(self.config_file)
            self.config = self.generateConfig(self.config)
        if "__default__" in self.config.Node.Nodes.keys():
            raise RuntimeError("Dun fucked up: __default__ node detected")
        if "__default__" in self.config.Node.Nodes.keys():
            raise RuntimeError("Dun fucked up: __default__ node detected")

        #Once Validated we've got no reason to hold on to the configobj structure

        self.nodes = []
        self.fleets = []

    def prepare(self, waits=False, *args, **kwargs):
        # Attempt Validation and construct the simulation from that config.
        try:
            baselogger.setLevel(LOGLEVELS.get(self.config.get("log_level"), logging.NOTSET))
            ch.setLevel(LOGLEVELS.get(self.config.get("log_level"), logging.NOTSET))
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
        """
        self.logger.info("Initialising Simulation %s, to run for %s steps" % (self.title, self.duration_intervals))
        for fleet in self.fleets:
            fleet.activate()
        if callback is not None:
            self.logger.info("Running with Callback: %s" % str(callback))
            Sim.startStepping()
        Sim.simulate(until=self.duration_intervals, callback=callback)
        self.logger.info("Finished Simulation at %s" % Sim.now())

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
        for node in self.nodes:
            pos = node.pos_log[:, :Sim.now()]
            vec = node.vec_log[:, :Sim.now()]
            pos = pos[np.isfinite(pos)].reshape(3, -1)
            vec = vec[np.isfinite(vec)].reshape(3, -1)

            positions.append(pos)
            vectors.append(vec)
            names.append(node.name)
        shape = self.environment.shape
        return positions, vectors, names, shape

    def validateConfig(self, config = None, final_check = False):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults

        NOTE: This does not verify if any of the functionality requested in the config is THERE
        Only that the config 'makes sense' as requested.

        I.e. does not check if particular modular behaviour exists or not.
        """

        #
        # GENERIC CONFIG ACQUISITION
        #
        if not isinstance(config, ConfigObj):
            config = ConfigObj(config, configspec=self.config_spec, stringify=True, interpolation=not final_check)
        else:
            self.logger.info("Skipping configobj for final validation")
        config_status = config.validate(validate.Validator(), copy= not final_check)

        if not config_status:
            # If config_spec doesn't match the input, bail
            raise ConfigError("Configspec doesn't match given input structure: %s" % config_status)

        return config

    def generateConfig(self, config):
        #
        # NODE CONFIGURATION
        #
        preconfigured_nodes_count = 0
        pre_node_names = []
        nodes_config = dict()
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
            nodes_config[node_name] = dict()
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

        # TODO Fleet implementation


        return node_list

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

    def postProcess(self, log=None, outputFile=None, displayFrames=None, dataFile=False, movieFile=False,
                    inputFile=None, xRes=1024, yRes=768, fps=24):
        """
        Performs output and positions generation for a given simulation
        """
        dpi = 80
        ipp = 80
        fig = plt.figure(dpi=dpi, figsize=(xRes / ipp, yRes / ipp))
        ax = axes3.Axes3D(fig)

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

        positions = []
        vectors = []
        names = []
        contributions = []
        shape = []
        if inputFile is not None:
            self.logger.info("Retrieving positions from file: %s" % inputFile)
            source = np.load(inputFile)
            positions = source['positions']
            vectors = source['vectors']
            names = source['names']
            shape = source['environment']
            assert len(positions) == len(names), 'Array Sizes don\'t match!'
        else:
            if log is None and inputFile is None:
                self.logger.info("Using default postprocessing log")
                log = self.environment.pos_log
            for node in self.nodes:
                positions.append(node.pos_log)
                vectors.append(node.vec_log)
                names.append(node.name)
                contributions.append(node.contributions_log)
            shape = self.environment.shape

        n_frames = len(positions[0][0])

        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], label=names[i])[0] for i, dat in enumerate(positions)]

        line_ani = AIETESAnimation(fig, updatelines, frames=int(n_frames), fargs=(positions, lines, displayFrames),
                                   interval=1000 / fps, repeat_delay=300, blit=True, )
        ax.legend()
        ax.set_xlim3d((0, shape[0]))
        ax.set_ylim3d((0, shape[1]))
        ax.set_zlim3d((0, shape[2]))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if outputFile is not None:
            filename = "%s.aietes" % outputFile
            if dataFile:
                self.logger.info("Writing datafile to %s" % filename)
                np.savez(filename,
                         positions=positions,
                         vectors=vectors,
                         names=names,
                         environment=self.environment.shape,
                         contributions=contributions
                         )
                co = ConfigObj(self.config, list_values=False)
                co.filename = "%s.conf" % filename
                co.write()
            if movieFile:
                self.logger.info("Writing animation to %s" % filename)
                self.save(line_ani,
                          filename=filename,
                          fps=fps,
                          codec='mpeg4',
                          clear_temp=True
                          )
        else:
            plt.show()

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
    sim = Simulation(config_file=options.config, title=options.title)

    if options.input is None:
        sim.prepare(sim_time=options.sim_time)
        if not options.noexecution:
            try:
                sim.simulate()
            except RuntimeError as exp:
                print(exp)
                print("Will try to postprocess anyway")

    if options.output:
        print("Storing output in %s" % options.title)
        sim.postProcess(inputFile=options.input, outputFile=options.title, dataFile=options.data,
                        movieFile=options.movie, fps=options.fps)

    if options.plot:
        sim.postProcess(inputFile=options.input, displayFrames=720)


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
        parser.add_option('-v', '--verbose', action='store_true',
                          default=False, help='verbose output')
        parser.add_option('-P', '--profile', action='store_true',
                          default=False, help='profiled execution')
        parser.add_option('-p', '--plot', action='store_true',
                          default=False, help='perform ploting')
        parser.add_option('-o', '--output', action='store_true',
                          default=False, help='store output')
        parser.add_option('-m', '--movie', action='store_true',
                          default=None, help='generate and store movie (this takes a long time)')
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
        if options.title is None:  # and no custom title
            if options.config is None:                                # If have no config
                options.title = dt.now().strftime('%Y-%m-%d-%H-%M-%S')  # use default title
            else:                                                     # if given config
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
