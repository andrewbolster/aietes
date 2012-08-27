import logging
import numpy as np
import scipy

from configobj import ConfigObj
import validate
import random

from Layercake import Layercake
import MAC, Net, Applications
from Environment import Environment
from Node import Node
import Behaviour

from Tools import *

import matplotlib; matplotlib.use('wxagg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3
import matplotlib.animation as ani
from operator import itemgetter,attrgetter

np.set_printoptions(precision=3)


class ConfigError(Exception):
    """
    Raised when a configuration cannot be validated through ConfigObj/Validator
    Contains a 'status' with the boolean dict representation of the error
    """
    def __init__(self,value):
        baselogger.critical("Invalid Config; Dying")
        self.status=value
    def __str__(self):
        return repr(self.status)

class Fleet(Sim.Process):
    """
    Fleets act initially as traffic managers for Nodes
    """
    def __init__(self,nodes,simulation):

        self.logger = baselogger.getChild("%s"%(self.__class__.__name__))
        self.logger.info("creating instance")
        Sim.Process.__init__(self,name="Fleet")
        self.nodes = nodes
        self.environment = simulation.environment
        self.simulation = simulation

    def activate(self):
        Sim.activate(self,self.lifecycle())
        for node in self.nodes:
            node.activate()


    def lifecycle(self):
        def allPassive():
            return all([n.passive() for n in self.nodes])
        self.logger.info("Initialised Node Lifecycle")
        while(True):
            yield Sim.waituntil, self, allPassive
            #self.logger.info("Fleet Step: %s EIG:(%s)"%(Sim.now(),self.environment.pointPlane()[0]))
            percent_now= ((100 * Sim.now()) / self.simulation.duration_intervals)
            if percent_now%1 == 0:
                self.logger.info("Fleet  %d%%: %s"%(percent_now,self.currentStats()))
            for node in self.nodes:
                Sim.reactivate(node)

    def currentStats(self):
        """
        Print Current Vector Statistics
        """
        avgHeading = np.array([0,0,0],dtype=np.float)
        fleetCenter = np.array([0,0,0],dtype=np.float)
        fleetWaypoints = []
        for node in self.nodes:
            avgHeading+=node.velocity
            fleetCenter+=node.position
            fleetWaypoints.append(node.behaviour.nextwaypoint.position)


        avgHeading/=float(len(self.nodes))
        fleetCenter/=float(len(self.nodes))

        maxDistance=np.float(0.0)
        maxDeviation=np.float(0.0)
        for node in self.nodes:
            maxDistance = max(maxDistance,distance(node.position,fleetCenter))
            v=node.velocity
            c= np.dot(avgHeading,v)/np.linalg.norm(avgHeading)/np.linalg.norm(v)
            maxDeviation = max(maxDeviation,np.arccos(c))

        commonheading = all(all(fleetWaypoints[0] == waypoint) for waypoint in fleetWaypoints)

        return("V:%s,W:%s,C:%s,D:%s,A:%s"%(avgHeading,commonheading,fleetCenter,maxDistance,maxDeviation))

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

        self.environment = self.configureEnvironment(self.config.Environment)
        self.nodes = self.configureNodes(self.config.Nodes)

        #Single Fleet to control all
        self.fleets.append(Fleet(self.nodes,self))

        # Set up 'join-like' operation for nodes
        self.move_flag = Sim.Resource(capacity=len(self.nodes))
        self.process_flag = Sim.Resource(capacity=len(self.nodes))

    def inner_join(self):
        """
        When all nodes have a move flag and none are processing
        """
        joined=self.move_flag.n == 0 and self.process_flag.n == len(self.nodes)
        if joined:
            self.logger.debug("Joined: %s,%s"%(self.move_flag.n,self.process_flag.n))
        return joined

    def outer_join(self):
        """
        When all nodes have a processing flag and none are moving
        """
        joined=self.move_flag.n == len(self.nodes) and self.process_flag.n == 0
        if joined:
            self.logger.debug("Joined: %s,%s"%(self.move_flag.n,self.process_flag.n))
        return joined

    def simulate(self):
        """
        Initiate the processed Simulation
        """
        self.duration_intervals = self.config.Simulation.sim_duration/self.config.Simulation.sim_interval
        self.logger.info("Initialising Simulation, to run for %s steps"%self.duration_intervals)
        for fleet in self.fleets:
            fleet.activate()

        Sim.simulate(until=self.duration_intervals)

    def reverse_node_lookup(self, uuid):
        """Return Node Given UUID
        """
        for n in self.nodes:
            if n.id == uuid:
                return n
        raise KeyError("Given UUID does not exist in Nodes list")


    def validateConfig(self,config_file='', configspec='defaults.conf'):
        """
        Generate valid configuration information by interpolating a given config
        file with the defaults
        """
        # GENERIC CONFIG CHECK
        config= ConfigObj(config_file,configspec=configspec)
        config_status = config.validate(validate.Validator(),copy=True)
        if not config_status:
            # If configspec doesn't match the input, bail
            raise ConfigError(config_status)

        config= dotdict(config.dict())
        naming_convention= config.Nodes.naming_convention
        # NODE CONFIGURATION
        if config.Nodes.count > len(naming_convention):
            # If the naming convention can't provide unique names, bail
            raise ConfigError("Not Enough Names in dictionary for number of nodes requested:%s/%s!"%(config.Nodes.count,len(naming_convention)))

        if bool(config.Nodes.node_names):
            # If given the correct number of names in the config file, do nothing
            if config.Nodes.count > len(config.Nodes.node_names):
                raise ConfigError("Not Enough Names in configfile for number of nodes requested!")
            assert int(config.Nodes.count) == len(config.Nodes.node_names)
        else:
            # Otherwise make up names from the naming_convention
            assert config.Nodes.count>=1, "No nodes configured"
            for n in range(config.Nodes.count):
                candidate_name= naming_convention[np.random.randint(0,len(naming_convention))]
                while candidate_name in config.Nodes.node_names:
                    candidate_name= naming_convention[np.random.randint(0,len(naming_convention))]
                config.Nodes.node_names.append(candidate_name)
                self.logger.info("Gave node %d name %s"%(n,candidate_name))

        # LAYERCAKE CONFIG CHECK
        try:
            config.mac_mod=getattr(MAC,str(config.MAC.protocol))
        except AttributeError:
            raise ConfigError("Can't find MAC: %s"%config.MAC.protocol)

        try:
            config.net_mod=getattr(Net,str(config.Network.protocol))
        except AttributeError:
            raise ConfigError("Can't find Network: %s"%config.Network.protocol)

        # APPLICATION DISTRIBUTION SANITY CHECK
        applications = config.Nodes.Application.protocol
        allocations = config.Nodes.Application.distribution
        node_count = config.Nodes.count
        try:

            if isinstance(applications, list):
                if isinstance(allocations, list):
                    app_checksum = len(applications) - len(allocations)
                    node_checksum = len(applications) - node_count
                    if app_checksum != 0 or node_checksum != 0:
                        raise ConfigError("Incorrect Applications specification: %s,[%s] for %d nodes"%(
                            applications,allocations,node_count))
                    else:
                        status_str = str(zip(applications,allocations))
                        self.logger.info("Assigned Vector Applications: %s"%status_str)
                        config.Nodes.app_mod=[getattr(Applications,str(app)) for app,n in zip(applications,allocations) for i in range(int(n))]
                        pass
                else:
                    raise ConfigError("List of Application but no list of distributions!: %s,[%s] for %d nodes"%(
                        applications,allocations,node_count))
            else:
                config.Nodes.app_mod=getattr(Applications,str(applications))
        except AttributeError:
            raise ConfigError("Can't find Application: %s"%applications)
        except Exception as e:
            raise e


        #Confirm
        self.logger.info("Built Config: %s"%str(config))
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

    def configureNodes(self,node_config,fleet=None):
        """
        Configure 'fleets' of nodes for simulation
        Fleets are purely logical in this case
        """
        nodelist = []
        #Configure specified nodes
        for node_id,nodeName in enumerate(node_config.node_names):
            config=self.vectorGen(
                nodeName,
                node_config,
                node_id
            )
            self.logger.debug("Generating node %s with config %s"%(nodeName,config))
            new_node=Node(
                nodeName,
                self,
                config
            )
            nodelist.append(new_node)

        #TODO Fleet implementation

        return nodelist

    def vectorGen(self,nodeName,node_config,node_id):
        """
        If a node is named in the configuration file, use the defined initial vector
        otherwise, use configured behaviour to assign an initial vector
        """
        try:# If there is an entry, use it
            vector = node_config.initial_node_vectors[nodeName]
            self.logger.info("Gave node %s a configured initial vector: %s"%(nodeName,str(vector)))
        except KeyError:
            #TODO add additional behaviours, eg random within enviroment; distribution around a point, etc
            gen_style = node_config.vector_generation
            if gen_style == "random":
                vector = self.environment.random_position()
                self.logger.debug("Gave node %s a random vector: %s"%(nodeName,vector))
            elif gen_style == "center":
                vector= self.environment.position_around()
                self.logger.debug("Gave node %s a center vector: %s"%(nodeName,vector))
            else:
                vector = [0,0,0]
                self.logger.debug("Gave node %s a zero vector from %s"%(nodeName,gen_style))
        assert len(vector)==3, "Incorrectly sized vector"

        # BEHAVIOUR CONFIG CHECK
        try:
            behave_mod=getattr(Behaviour,str(node_config.Behaviour.protocol))
            self.logger.debug("Using behaviour: %s"%behave_mod)
        except AttributeError:
            raise ConfigError("Can't find Behaviour: %s"%node_config.Behaviour.protocol)

        # APPLICATIONS CONFIG CHECK
        try:
            if isinstance(node_config.app_mod,list):
                app_mod = node_config.app_mod[node_id]
            else:
                app_mod = node_config.app_mod
            self.logger.info("Using Application: %s"%app_mod)
        except AttributeError:
            raise ConfigError("Can't find Behaviour: %s"%node_config.Behaviour.protocol)
        
        config = dotdict({  'vector':vector,
                            'max_speed':node_config.max_speed,
                            'max_turn':node_config.max_turn,
                            'behave_mod':behave_mod,
                            'app_mod': app_mod,
                            'Behaviour':node_config.Behaviour
                        })
        return config

    def postProcess(self,log=None,outputFile=None,displayFrames=None,dataFile=False,movieFile=False,inputFile=None,xRes=1024,yRes=768,fps=24):
        """
        Performs output and data generation for a given simulation
        """
        dpi=80
        ipp=80
        fig = plt.figure(dpi=dpi, figsize=(xRes/ipp,yRes/ipp))
        ax = axes3.Axes3D(fig)
        
        def updatelines(i,data,lines,displayFrames):
            """
            Update the currently displayed line data
            data contains [x,y,z],[t] data for each vector
            displayFrames configures the display cache size
            """
            if isinstance(displayFrames,int):
                j=max(i-displayFrames,0)
            else:
                j=0
            for line,dat in zip(lines,data):
                line.set_data(dat[0:2, j:i])         #x,y axis
                line.set_3d_properties(dat[2, j:i])  #z axis
            return lines

        data = []
        names = []
        shape = []
        if inputFile is not None:
            self.logger.info("Retrieving data from file: %s"%inputFile)
            (data,names,shape) = (np.load(inputFile).items())[0][1]
            assert len(data)==len(names), 'Array Sizes don\'t match!'
        else:
            if log is None and inputFile is None:
                self.logger.info("Using default postprocessing log")
                log=self.environment.pos_log
            for node in self.nodes:
                data.append(node.pos_log)
                names.append(node.name)
            shape=self.environment.shape

        n_frames= len(data[0][0])

        lines = [ax.plot(dat[0, 0:1], dat[1,0:1], dat[2, 0:1],label=names[i])[0] for i,dat in enumerate(data) ]

        line_ani = ani.FuncAnimation(fig, updatelines, frames=int(n_frames), fargs=(data, lines, displayFrames), interval=1000/fps, repeat_delay=300,  blit=True,)
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
                np.savez(filename,(data,names,self.environment.shape))
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

