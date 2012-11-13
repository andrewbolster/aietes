import logging
import numpy as np
import scipy

from configobj import ConfigObj
import validate
import random

from Layercake import Layercake
from Environment import Environment
from Fleet import Fleet
from Node import Node
import Behaviour

from Tools import *

import matplotlib; matplotlib.use('wxagg')
from matplotlib import animation as MPLanimation

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3
from operator import itemgetter,attrgetter

from pprint import pformat

np.set_printoptions(precision=3)


class Simulation():
    """
    Defines a single simulation
    """
    def __init__(self,config_file=''):
        self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
        self.logger.info("creating instance from %s"%config_file)
        self.config_file=config_file
        self.nodes=[]
        self.fleets=[]

    def prepare(self):
        #Attempt Validation and construct the simulation from that config.
        try:
            self.config=self.validateConfig(self.config_file)
            baselogger.setLevel(LOGLEVELS.get(self.config.log_level,logging.NOTSET))
        except ConfigError as err:
            self.logger.error("Error in configuration, cannot continue: %s"%err)
            raise SystemExit(1)

        #Initialise simulation environment and configure a global channel event
        Sim.initialize()
        self.channel_event = Sim.SimEvent(self.config.Simulation.channel_event_name)
        self.duration_intervals = self.config.Simulation.sim_duration/self.config.Simulation.sim_interval

        self.environment = self.configureEnvironment(self.config.Environment)
        self.nodes = self.configureNodes()

        #Single Fleet to control all
        self.fleets.append(Fleet(self.nodes,self))

        # Set up 'join-like' operation for nodes
        self.move_flag = Sim.Resource(capacity=len(self.nodes))
        self.process_flag = Sim.Resource(capacity=len(self.nodes))

    def simulate(self):
        """
        Initiate the processed Simulation
        """
        self.logger.info("Initialising Simulation, to run for %s steps"%self.duration_intervals)
        for fleet in self.fleets:
            fleet.activate()

        Sim.simulate(until=self.duration_intervals)
    def inner_join(self):
        """
        When all nodes have a move flag and none are processing
        """
        joined=self.move_flag.n == 0 and self.process_flag.n == len(self.nodes)
        if joined and debug:
            self.logger.debug("Joined: %s,%s"%(self.move_flag.n,self.process_flag.n))
        return joined

    def outer_join(self):
        """
        When all nodes have a processing flag and none are moving
        """
        joined=self.move_flag.n == len(self.nodes) and self.process_flag.n == 0
        if joined and debug:
            self.logger.debug("Joined: %s,%s"%(self.move_flag.n,self.process_flag.n))
        return joined

    def reverse_node_lookup(self, uuid):
        """Return Node Given UUID
        """
        for n in self.nodes:
            if n.id == uuid:
                return n
        raise KeyError("Given UUID does not exist in Nodes list")


    def validateConfig(self,config_file='', configspec='configs/default.conf'):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults

        NOTE: This does not verify if any of the functionality requested in the config is THERE
        Only that the config 'makes sense' as requested.

        I.e. does not check if particular modular behaviour exists or not.
        """

        ##############################
        # GENERIC CONFIG ACQUISITION
        ##############################

        config= ConfigObj(config_file,configspec=configspec)
        config_status = config.validate(validate.Validator(),copy=True)

        if not config_status:
            # If configspec doesn't match the input, bail
            raise ConfigError("Configspec doesn't match given input structure: %s" % config_status)

        config= dotdict(config.dict())


        ##############################
        # NODE CONFIGURATION
        ##############################
        nodes_preconfigured = 0
        pre_node_names=[]
        nodes_config = dict()
        node_default_config = config.Node.Nodes.pop('__default__')
        # Add the stuff we know whould be there...
        self.logger.debug("Default Node Config: %s" % pformat(node_default_config))
        node_default_config.update(
            #TODO import PHY,Behaviour, etc into the node config?
        )


        ###
        # Check if there are individually configured nodes
        if isinstance(config.Node.Nodes, dict) and len(config.Node.Nodes) > 0:
            ###
            #There Are Detailed Config Instances
            nodes_preconfigured = len(config.Node.Nodes)
            self.logger.info("Have %d nodes from config: %s" % (
                                    nodes_preconfigured,
                                    nodes_config)
            )
            pre_node_names = config.Node.Nodes.keys()

        ###
        # Check and generate application distribution
        #   i.e. app = ["App A","App B"]
        #        dist = [ 4, 5 ]
        try:
            app = node_default_config.Application.protocol
            dist = node_default_config.Application.distribution
            nodes_count = config.Node.count
        except AttributeError as e:
            self.logger.error("Error:%s"%e)
            self.logger.info("%s" % pformat(node_default_config))
            raise ConfigError("Something is badly wrong")

        # Boundary checks:
        #   len(app)==len(dist)
        #   len(app) % nodes_count-nodes_preconfigured = 0
        self.logger.debug("App:%s,Dist:%s" % (app,dist))
        if isinstance(app,list) and isinstance(dist, list):
            if len(app) == len(dist) and (nodes_count-nodes_preconfigured) % len(app) == 0:
                applications = [str(a)
                                for a,n in zip(app,dist)
                                    for i in range(int(n))
                               ]
                self.logger.info("Distributed Applications:%s" % applications)
            else:
                raise ConfigError(
                    "Application / Distribution mismatch"
                )
        else:
            applications = [str(app) for i in range(int(nodes_count-nodes_preconfigured))]
            self.logger.info("Using Application:%s" % applications)

        ###
        # Generate Names for any remaining auto-config nodes
        auto_node_names = nameGeneration(
            count = config.Node.count - nodes_preconfigured,
            naming_convention = config.Node.naming_convention
        )
        node_names = auto_node_names + pre_node_names

        # Give defaults to all
        for node_name in node_names:
            # Bare Dict/update instead of copy()
            nodes_config[node_name]=dict()
            nodes_config[node_name].update(node_default_config)

        # Give auto-config default
        for node_name, node_app in zip(auto_node_names,applications):
            # Add derived application
            nodes_config[node_name]['app']=str(node_app)

        # Overlay Preconfigured with their own settings
        for node_name, node_config in config.Node.Nodes.items():
            # Import the magic!
            nodes_config[node_name].update(node_config)

        ###
        # Generate Per-Vector Node initialisation configs
        for node_name,node_config in nodes_config.items():
            self.logger.info("[%s]: %s" % (node_name,node_config))

        ##############################
        #Confirm
        ##############################
        config.Node.Nodes.update(nodes_config)
        self.logger.debug("Built Config: %s"%pformat(config))

        return config

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
        ##############################
        #Configure specified nodes
        ##############################
        for node_name,config in self.config.Node.Nodes.items():
            self.logger.debug("Generating node %s with config %s"%(node_name,config))
            new_node=Node(
                node_name,
                self,
                config,
                vector = self.vectorGen(node_name,config)
            )
            node_list.append(new_node)

        #TODO Fleet implementation

        return node_list

    def vectorGen(self,node_name,node_config):
        """
        If a node is named in the configuration file, use the defined initial vector
        otherwise, use configured behaviour to assign an initial vector
        """
        try:# If there is an entry, use it
            vector = node_config['initial_vector']
            self.logger.info("Gave node %s a configured initial vector: %s"%(node_name,str(vector)))
        except KeyError:
            gen_style = node_config['position_generation']
            if gen_style == "random":
                vector = self.environment.random_position()
                self.logger.debug("Gave node %s a random vector: %s"%(node_name,vector))
            elif gen_style == "center":
                vector= self.environment.position_around()
                self.logger.debug("Gave node %s a center vector: %s"%(node_name,vector))
            else:
                vector = [0,0,0]
                self.logger.debug("Gave node %s a zero vector from %s"%(node_name,gen_style))
        assert len(vector)==3, "Incorrectly sized vector"

        return vector

    def postProcess(self,log=None,outputFile=None,displayFrames=None,dataFile=False,movieFile=False,inputFile=None,xRes=1024,yRes=768,fps=24):
        """
        Performs output and positions generation for a given simulation
        """
        dpi=80
        ipp=80
        fig = plt.figure(dpi=dpi, figsize=(xRes/ipp,yRes/ipp))
        ax = axes3.Axes3D(fig)
        
        def updatelines(i,positions,lines,displayFrames):
            """
            Update the currently displayed line positions
            positions contains [x,y,z],[t] positions for each vector
            displayFrames configures the display cache size
            """
            if isinstance(displayFrames,int):
                j=max(i-displayFrames,0)
            else:
                j=0
            for line,dat in zip(lines,positions):
                line.set_data(dat[0:2, j:i])         #x,y axis
                line.set_3d_properties(dat[2, j:i])  #z axis
            return lines

        positions = []
        vectors= []
        names = []
        shape = []
        if inputFile is not None:
            self.logger.info("Retrieving positions from file: %s"%inputFile)
            source = np.load(inputFile)
            positions = source['positions']
            vectors = source['vectors']
            names = source['names']
            shape = source['environment']
            assert len(positions)==len(names), 'Array Sizes don\'t match!'
        else:
            if log is None and inputFile is None:
                self.logger.info("Using default postprocessing log")
                log=self.environment.pos_log
            for node in self.nodes:
                positions.append(node.pos_log)
                vectors.append(node.vec_log)
                names.append(node.name)
            shape=self.environment.shape

        n_frames= len(positions[0][0])

        lines = [ax.plot(dat[0, 0:1], dat[1,0:1], dat[2, 0:1],label=names[i])[0] for i,dat in enumerate(positions) ]

        line_ani = AIETESAnimation(fig, updatelines, frames=int(n_frames), fargs=(positions, lines, displayFrames), interval=1000/fps, repeat_delay=300,  blit=True,)
        ax.legend()
        ax.set_xlim3d((0,shape[0]))
        ax.set_ylim3d((0,shape[1]))
        ax.set_zlim3d((0,shape[2]))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if outputFile is not None:
            if dataFile:
                filename = "dat-%s"%outputFile
                self.logger.info("Writing datafile to %s"%filename)
                np.savez(filename,
                         positions=positions,
                         vectors=vectors,
                         names=names,
                         environment=self.environment.shape
                        )
                co=ConfigObj(self.config)
                co.filename=outputFile+'.conf'
                co.write()
            if movieFile:
                filename = "ani-%s"%outputFile
                self.logger.info("Writing animation to %s"%filename)
                save(line_ani,
                     filename=filename,
                     fps=fps,
                     codec='mpeg4',
                     clear_temp=True
                    )
        else:
            plt.show()


    def deltaT(self,now,then):
        """
        Time in seconds between two simulation times
        """
        return (now-then)*self.config.Simulation.sim_interval

class AIETESAnimation(MPLanimation.FuncAnimation):

    def save(self, filename, fps=5, codec='libx264', clear_temp=True,
        frame_prefix='_tmp'):
        '''
        Saves a movie file by drawing every frame.

        *filename* is the output filename, eg :file:`mymovie.mp4`

        *fps* is the frames per second in the movie

        *codec* is the codec to be used,if it is supported by the output method.

        *clear_temp* specifies whether the temporary image files should be
        deleted.

        *frame_prefix* gives the prefix that should be used for individual
        image files.  This prefix will have a frame number (i.e. 0001) appended
        when saving individual frames.
        '''
        # Need to disconnect the first draw callback, since we'll be doing
        # draws. Otherwise, we'll end up starting the animation.
        if self._first_draw_id is not None:
            self._fig.canvas.mpl_disconnect(self._first_draw_id)
            reconnect_first_draw = True
        else:
            reconnect_first_draw = False

        fnames = []
        # Create a new sequence of frames for saved data. This is different
        # from new_frame_seq() to give the ability to save 'live' generated
        # frame information to be saved later.
        # TODO: Right now, after closing the figure, saving a movie won't
        # work since GUI widgets are gone. Either need to remove extra code
        # to allow for this non-existant use case or find a way to make it work.
        for idx,data in enumerate(self.new_saved_frame_seq()):
            #TODO: Need to see if turning off blit is really necessary
            self._draw_next_frame(data, blit=False)
            fname = '%s%04d.png' % (frame_prefix, idx)
            fnames.append(fname)
            self._fig.savefig(fname)

        _make_movie(self,filename, fps, codec, frame_prefix, cmd_gen=mencoder_cmd)

        #Delete temporary files
        if clear_temp:
            import os
            for fname in fnames:
                os.remove(fname)

        # Reconnect signal for first draw if necessary
        if reconnect_first_draw:
            self._first_draw_id = self._fig.canvas.mpl_connect('draw_event',
                self._start)

    def ffmpeg_cmd(self, fname, fps, codec, frame_prefix):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie
        return ['ffmpeg', '-y', '-r', str(fps),
                '-b', '1800k', '-i','%s%%04d.png' % frame_prefix,
                '-vcodec', codec, '-vpre','slow','-vpre','baseline',
                "%s.mp4"%fname]

    def mencoder_cmd(self, fname, fps, codec, frame_prefix):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return ['mencoder',
                   '-nosound',
                   '-quiet',
                   '-ovc', 'lavc',
                   '-lavcopts',"vcodec=%s"%codec,
                   '-o', "%s.mp4"%fname,
                   '-mf', 'type=png:fps=24', 'mf://%s%%04d.png'%frame_prefix]


    def _make_movie(self, fname, fps, codec, frame_prefix, cmd_gen=None):
            # Uses subprocess to call the program for assembling frames into a
            # movie file.  *cmd_gen* is a callable that generates the sequence
            # of command line arguments from a few configuration options.
            from subprocess import Popen, PIPE
            if cmd_gen is None:
                cmd_gen = ffmpeg_cmd
            command = cmd_gen(self,fname, fps, codec, frame_prefix)
            print command
            proc = Popen(command, shell=False,
                stdout=PIPE, stderr=PIPE)
            proc.wait()

