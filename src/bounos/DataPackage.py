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
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

__author__ = 'bolster'
import sys
import traceback
import logging

from scipy.spatial.distance import pdist, squareform

# used to reconstitute config from NP object
from ast import literal_eval

import numpy as np
import pandas as pd


from aietes.Tools import mag, add_ndarray_to_set, unext, validateConfig, results_file
from configobj import ConfigObj


class DataPackage(object):

    """
    Data Store for simulation results
    Replicates a Numpy Array as is directly accessible as an n-vector
    [x,y,z,t] array for n simulated nodes positions over time. Additional
    information is queriable through the object.
    """
    _attrib_map = {'p': 'positions',
                   'v': 'vectors',
                   'plr': 'plr',
                   'rssi': 'rssi',
                   'data_rate': 'data_rate',
                   'delay': 'delay',
                   'names': 'names',
                   'contributions': 'contributions',
                   'achievements': 'achievements',
                   'environment': 'environment',
                   'config': 'config',
                   'title': 'title',
                   'waypoints': 'waypoints',
                   'drift_positions': 'drift_positions',
                   'intent_positions': 'ecea_positions',
                   'ecea_positions': 'ecea_positions',
                   'additional': 'additional',
                   'comms': 'comms'
                   }

    version = 1.0

    def _handle_mapping_exceptions_(self, sink, source, source_filename, exp):
        """
        Attempts to extract DataPackage data from a source dict
        """
        logging.debug("Caught exception on sink:%s source:%s (%s) attempting to recover"
                      % (sink, source, exp))
        # Per Attrib Fixes in the case of failure
        if sink is "title":
            # If simulation title not explicitly given, use the filename -npz
            self.title = ""
        elif sink is "achievements":
            # If using pre-achievements datapackage, turn achievement stats off
            # by setting achievements to none
            logging.debug("Pre-Achievements Datapackage")
            self.achievements = None
        elif sink is "waypoints":
            # If using pre-waypoints datapackage, turn waypoints off
            logging.debug("Pre-Waypoints Datapackage")
            self.waypoints = None
        elif sink is "config":
            # If config file is not listed, look for one in the same dir with
            # the same name
            potential_config_file = unext(source_filename) + ".conf"
            logging.error("Potential Config %s" % potential_config_file)
            self.config = validateConfig(potential_config_file)
        elif sink is "drift_positions":
            logging.debug("Non-drifting Datapackage")
            self.drifting = False
        elif sink is "intent_positions":
            """ ECEA_Positions used to be called Intent
            """
            logging.debug("Non-ECEA Datapackage")
            if not hasattr(self, 'ecea'):
                self.ecea = False
        elif sink is "ecea_positions":
            logging.debug("Non-ECEA Datapackage")
            if not hasattr(self, 'ecea'):
                self.ecea = False
        elif sink is "additional":
            self.additional = None
        # This method may fail miserable is sink and source are named
        # differently
        elif sink in ['data_rate', 'plr', 'delay', 'rssi']:
            setattr(self, sink, None)

        else:
            raise ValueError("Can't find %s(%s) in source" % (source, sink))

    def __init__(self, source=None, *args, **kwargs):
        """
        Raises IOError on load failure
        """
        try:
            if source is not None:
                if isinstance(source, list):
                    raise ValueError("Source file shoult be singular, not a list {}".format(source))
                else:
                    kwargs.update(**np.load(source))

            """ Attempt to fix sink source mapping between file stores and runtime stores"""
            for sink_attrib, source_attrib in self._attrib_map.iteritems():
                try:
                    self.__dict__[sink_attrib] = kwargs[sink_attrib]
                except (AttributeError, KeyError) as exp:
                    self._handle_mapping_exceptions_(
                        sink_attrib, source_attrib, "kwargs", exp)
        except (AttributeError, KeyError) as exp:
            raise ValueError("Inappropriate / incomplete source of type {} : {} missing" .format(
                type(source),
                str([attr for attr in self._attrib_map.keys()
                     if not kwargs.has_key(attr)])
            )
            ),  None, traceback.print_tb(sys.exc_info()[2])

        self.tmax = int(kwargs.get("tmax", len(self.p[0][0])))
        self.n = len(self.p)

        if self.tmax > self.p.shape[2]:
            logging.warn("Missized datapackage({} vs {}): hopefully a mission success: resetting tmax".format(
                self.tmax, self.p.shape))
            self.tmax = self.p.shape[2]

        # Fixes for Datatypes
        self.config = literal_eval(str(self.config))
        if isinstance(self.names, np.ndarray):
            self.names = self.names.tolist()

        # Convoluted logic to by DEFAULT assume drifting, unless it's set by
        # the failing case in _handle
        if not hasattr(self, 'drifting'):
            self.drifting = True
        if not hasattr(self, 'ecea'):
            self.ecea = True

    def pad_time(self, tmax, val=0.0):
        """
        In some cases it's necessary to mis datapackage data, which can be difficult with different length
        simulations.

        `pad_time` extends the arrays in the package that are related to timing (pretty much everything)
        Does NOT modify `self.tmax`

        NOT TESTED
        """
        if tmax > self.tmax:
            for datum in ['p', 'v', 'drift_positions']:
                pad_val = np.array(3)
                pad_val.fill(val)
                orig_shape = self.__dict__[datum].shape
                self.__dict__[datum] = np.asarray([
                    np.pad(
                        self.__dict__[datum], (pad_val, tmax - self.tmax), mode="constant")
                ])
        elif tmax < self.tmax:
            raise ValueError("You can't trim my package!")
        else:
            # Same tmax, no problem
            pass

    def update(self, p=None, v=None, names=None, environment=None, **kwargs):
        logging.debug("Updating from tmax %d" % self.tmax)

        if all(x is not None for x in [p, v, names, environment]):
            self.p = p
            self.v = v
            self.names = names
            self.environment = environment
        else:
            raise ValueError("Can't work out what the hell you want!")
        self.tmax = len(self.p[0][0])
        self.n = len(self.p)

    def write(self, filename=None, track_mat=False):
        logging.info(
            "Writing datafile to {}.npz".format(results_file(filename)))

        data = {i: self.__dict__[
            i] for i in self._attrib_map.keys() if self.__dict__.has_key(i)}
        data['filename'] = "{}.npz".format(filename)

        np.savez(filename,
                 **(data)
                 )
        co = ConfigObj(self.config, list_values=False)
        co.filename = "%s.conf" % filename
        co.write()
        if track_mat:
            self.export_track_mat(filename)
        return ("%s.npz" % filename, co.filename)

    # Data has the format:
    # [n][x,y,z][t]

    def getBehaviourDict(self):
        behaviour_set = set()
        config_dict = self.config
        try:
            node_config_dict = config_dict['Node']['Nodes']
        except ValueError:
            print(config_dict)
            raise
        behaviours = {}

        if node_config_dict:
            for name, node in node_config_dict.iteritems():
                n_bev = node['Behaviour']['protocol']
                behaviour_set.add(n_bev)
                if n_bev in behaviours:
                    behaviours[n_bev].append(name)
                else:
                    behaviours[n_bev] = [name]
        return behaviours

    def getExtent(self):
        """
        Return the 3D Limits of the experiment
        """
        return zip(np.min(np.min(self.p, axis=0), axis=1), np.max(np.max(self.p, axis=0), axis=1))

    def achievement_statistics(self):
        """
        Generate Achievements Statistics
                'achievements':{
                    'pernode':{nodename:{time:tuple(achievement positio)}
                    '}
        """
        ach_stat = {}
        ach_stat = {'pernode': {}}

        ach_pos = list()
        max_ach = 0
        min_ach = np.inf
        max_acher = None
        min_acher = None

        tot_achs = 0
        for i, name in enumerate(self.names):
            n_ach_stat = {}
            n_ach = self.achievements[i]
            for time in np.asarray(n_ach.nonzero()).flatten().tolist():
                n_ach_stat[time] = n_ach[time]
                ach_pos = add_ndarray_to_set(n_ach[time][0][0], ach_pos)
            ach_stat['pernode'][name] = n_ach_stat

            achs_count = len(n_ach_stat)
            if achs_count > max_ach:
                max_acher = name
                max_ach = achs_count
            if achs_count < min_ach:
                min_acher = name
                min_ach = achs_count
            tot_achs += achs_count

        top_guns = []
        for name, n_ach_stat in ach_stat['pernode'].iteritems():
            achs_count = len(n_ach_stat)
            if achs_count == max_ach:
                top_guns.append(name)

        ach_stat['max_ach'] = max_ach
        ach_stat['min_ach'] = min_ach
        ach_stat['ach_pos'] = ach_pos
        ach_stat['avg_completion'] = float(tot_achs) / len(ach_stat['pernode'])
        ach_stat['percent_completion'] = float(
            len(top_guns)) / len(ach_stat['pernode'])

        return ach_stat

    def package_statistics(self):
        """
        Generate General Package Statistics, i.e. full simulation statistics
        Returns:
            A dict containing:
                'motion':{}

        """

        stats = {}
        """Achievement Statistics"""
        if self.achievements is not None:
            stats['achievements'] = self.achievement_statistics()

        """Volume Statistics"""
        # TODO The maths for this is insane....

        """Motion Statistics"""
        mot_stat = {}
        # Standard Variation of the Internode Distance Average would represent
        # the variability of the fleet
        mot_stat["std_of_INDA"] = np.std(
            [self.inter_distance_average(t) for t in xrange(self.tmax)])
        mot_stat["std_of_INDD"] = np.std(
            [self.position_matrix(t) / self.inter_distance_average(t) for t in xrange(self.tmax)])
        # Fleet speed (i.e. total distance covered) would represent comparative
        # efficiency
        mot_f_distance = np.sum(map(mag, self.v))
        mot_stat["fleet_distance"] = mot_f_distance
        mot_stat["fleet_efficiency"] = (mot_f_distance / self.tmax) / self.n

        stats['motion'] = mot_stat

        return stats

    def position_of(self, node, time):
        """
        Query the data set for the x,y,z position of a node at a given time
        """
        return self.p[node, :, time]

    def heading_of(self, node, time):
        """
        Query the data set for the x,y,z vector of a node at a given time
        """
        return self.v[node, :, time]

    def position_slice(self, time):
        """
        Query the dataset for the [n][x,y,z] position list of all nodes at a given time
        """
        return self.p[:, :, time]

    def heading_slice(self, time):
        """
        Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
        """
        return self.v[:, :, time]

    def position_slice_of(self, node):
        """
        Query the dataset for the [n][x,y,z] position list of all nodes at a given time
        """
        try:
            return self.p[node, :, :]
        except IndexError as e:
            logging.debug("Position Query for n:%d @ all for position shape %s" % (
                node, self.p[node].shape))
            raise e

    def heading_slice_of(self, node):
        """
        Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
        """
        try:
            return self.v[node, :, :]
        except IndexError as e:
            logging.debug("Heading Query for n:%d @ all for heading shape %s" % (
                node, self.v[node].shape))
            raise e

    def heading_mag_range(self):
        """
        Returns an array of average heading magnitudes across the dataset
        i.e. the average speed for each node in the fleet at each time

        """
        magnitudes = [sum(map(mag, self.heading_slice(time))) / self.n
                      for time in range(self.tmax)
                      ]
        return magnitudes

    def average_heading_mag_range(self):
        """
        Returns an array of average heading magnitudes across the dataset
        i.e. the average speed for each node in the fleet at each time

        """
        magnitudes = [mag(self.average_heading(time))
                      for time in range(self.tmax)
                      ]
        return magnitudes

    def heading_stddev_range(self):
        """
        Returns an array of overall heading stddevs across the dataset

        """
        deviations = [np.std(
            self.deviation_from_at(self.average_heading(time), time)
        )
            for time in range(self.tmax)
        ]
        return deviations

    def average_position(self, time):
        """
        Generate the average position (center) for the fleet at a given
        timeslice

        :param time: time index to calculate at
        :type int

        :raises ValueError
        """
        if not (0 <= time <= self.tmax):
            raise ValueError(
                "Time must be in the range of the dataset: %s, %s" % (time, self.tmax))

        return np.average(self.position_slice(time), axis=0)

    def average_heading(self, time):
        """
        Generate the average heading for the fleet at a given timeslice
        :param time: time index to calculate at
        :type int

        :raises ValueError
        """
        if not (0 <= time <= self.tmax):
            raise ValueError(
                "Time must be in the range of the dataset: %s, %s" % (time, self.tmax))

        return np.average(self.heading_slice(time), axis=0)

    def deviation_from_at(self, heading, time):
        """
        Return a one dimensional list of the linear deviation from
        each node to a given heading

        :param time: time index to calculate at
        :type int

        :param heading: (x,y,z) heading to compare against
        :type tuple
        """
        return map(
            lambda v: mag(
                np.array(v) - np.array(heading)
            ), self.heading_slice(time)
        )

    def sphere_of_positions(self, time):
        """
        Return the x,y,z,r configuration of the fleet encompassing
        it's location and size

        :param time: time index to calculate at
        :type int

        """
        x, y, z = self.average_position(time)
        max_r = max(self.distances_from_at((x, y, z), time))

        return x, y, z, max_r

    def distances_from_at(self, position, time):
        """
        Return a one dimensional list of the linear distances from
        each node to a given position

        :param time: time index to calculate at
        :type int

        :param position: (x,y,z) position to compare against
        :type tuple
        """
        return map(
            lambda p: mag(
                np.array(p) - np.array(position)
            ), self.position_slice(time)
        )

    def distances_from_average_at(self, time):
        """
        Return a one dimensional list of the linear distances from
        each node to the current fleet average point
        """
        return self.distances_from_at(self.average_position(time), time)

    def distance_from_average_stddev_range(self):
        """
        Returns an array of overall stddevs across the dataset
        """
        return [np.std(self.distances_from_average_at(time)) for time in range(self.tmax)]

    def position_matrix(self, time):
        """
        Returns a positional matrix of the distances between all the nodes at a given time.
        """
        return squareform(
            pdist(
                self.position_slice(time)
            )
        )

    def contribution_slice(self, node, time):
        """
        Query the dataset for the [n][b][x,y,z] contribution list of all nodes at a given time
        """
        try:
            return self.contributions[node, time]
        except IndexError as e:
            try:
                logging.debug(
                    "Contribution Query for n:%d @ all for position shape %s" % (node, self.contributions[node].shape))
                raise e
            except AttributeError as e:
                logging.error(
                    "Ok, the debug message failed, this looks like a corrupted file so I'll just shut off the contrib ")

    def inter_distance_average(self, time):
        """
        Returns the average distance between nodes
        """
        return np.average(self.position_matrix(time))

    def drift_error(self, source="drift"):
        """
        Returns the drift errors for each node
        in real meters
        """
        if self.drifting:
            if source == "drift":
                return np.linalg.norm(self.drift_positions - self.p, axis=1)
            elif source == "intent":
                return np.linalg.norm(self.ecea_positions - self.p, axis=1)
            else:
                raise ValueError("Invalid Drift Source: {}".format(source))
        else:
            raise ValueError("Not Drifting")

    def drift_RMS(self, source="drift"):
        """
        Returns the RMS of drift across all nodes over time
        """
        return np.sqrt(np.sum(self.drift_error(source), axis=0) / self.n)

    def ecea_error(self, periodic=True):
        """
        Nasty hacky dirty stuff to get the RMS statistics out of the 'additional' section across executions
        :return:
        """
        period = np.asarray(
            map(lambda x: x['params']['Delta'], self.additional))
        if np.all(period == period[0]):
            period = period[0]
        if periodic:
            truth = np.asarray(map(lambda x: x['true'], self.additional))
            drift = np.asarray(map(lambda x: x['drift'], self.additional))
            estimate = np.asarray(
                map(lambda x: x['estimate'], self.additional))
        else:
            truth = self.p
            drift = self.drift_positions
            estimate = self.ecea_positions
            period = -period

        drift_err = np.linalg.norm(drift - truth, axis=1)
        estimate_err = np.linalg.norm(estimate - truth, axis=1)

        return (drift_err, estimate_err, self.n, period)

    def export_track_mat(self, filename=None):
        """
        Export the True, Drift and ECEA (if available) for each node into a <title>.tracks.mat file,
        {   true_positions:
            drift_positions:
            ecea_positions
        }
        :return:
        """

        from scipy.io import savemat
        if filename is None:
            filename = self.title
        path = "{}.track".format(filename)
        data = {
            'true_positions': self.p,
        }
        if hasattr(self, "drift_positions"):
            data.update({'drift_positions': self.drift_positions})
        if hasattr(self, "ecea_positions"):
            data.update({'ecea_positions': self.ecea_positions})
        savemat(path, data)

    def generate_animation(self, filename=None, fps=24, gif=False, movieFile=True, xRes=1024, yRes=768, extent=True, displayFrames=None):
        """

        :param filename: Defaults to Title if not given; appended with mp4 / gif
        :param fps:
        :param gif: Bool to build gif
        :param movieFile: bool to build movie
        :param xRes: Frame Size
        :param yRes: Frame Size
        :param extent: Automatically zoom to the extent of operation rather than the environment
        :return:
        """
        import matplotlib

        matplotlib.use('Agg')
        from Animation import AIETESAnimation
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.axes3d as axes3

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

        if filename is None:
            filename = self.title

        return_dict = {}

        dpi = 80
        ipp = 80
        n_frames = self.tmax
        fig = plt.figure(dpi=dpi, figsize=(xRes / ipp, yRes / ipp))
        ax = axes3.Axes3D(fig)
        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], label=self.names[i])[0] for i, dat in
                 enumerate(self.p)]
        ax.legend()

        if extent is True:
            extent = tuple(zip(
                np.min(np.min(self.p, axis=0), axis=1), np.max(np.max(self.p, axis=0), axis=1)))
            (lx, rx), (ly, ry), (lz, rz) = extent
            x_width = abs(lx - rx)
            y_width = abs(ly - ry)
            z_width = abs(lz - rz)
            width = max(x_width, y_width, z_width) * 1.2

            avg = np.average(np.average(self.p, axis=0), axis=1)
            ax.set_xlim3d((avg[0] - (width / 2), avg[0] + (width / 2)))
            ax.set_ylim3d((avg[1] - (width / 2), avg[1] + (width / 2)))
            ax.set_zlim3d((avg[2] - (width / 2), avg[2] + (width / 2)))
        else:
            ax.set_xlim3d((0, self.environment[0]))
            ax.set_ylim3d((0, self.environment[1]))
            ax.set_zlim3d((0, self.environment[2]))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        line_ani = AIETESAnimation(fig, updatelines, frames=int(n_frames), fargs=(self.p, lines, displayFrames),
                                   interval=1000 / fps, repeat_delay=300, blit=True, )
        if movieFile:
            logging.info("Writing animation to %s.mp4" % filename)
            line_ani.save(
                filename=filename,
                fps=fps,
                codec='mpeg4',
                clear_temp=True
            )
            return_dict['ani_file'] = "%s.mp4" % filename
        if gif:
            logging.info("Writing animation to %s.gif" % filename)
            from matplotlib import animation as MPLanimation
            from matplotlib import verbose

            verbose.level = "debug"

            return_dict['ani_file'] = "%s.gif" % filename
            MPLanimation.FuncAnimation.save(line_ani, filename=return_dict['ani_file'], extra_args="-colors 8",
                                            writer='imagemagick', bitrate=-1, fps=fps)
        return plt, return_dict

    def plot_axes_views(self, res=120):
        """
        Plot a physical overview of the fleet and it's motion.
        :param res: timing resolution multiplier in seconds (default 'every 2 minutes of simulated time')
        :return: f
        """
        import matplotlib.pyplot as plt
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        f.suptitle("Node Layout and mobility for {}".format(self.title))
        for n, node_p in enumerate(self.p):
            x,y,z=node_p[:,0]

            ax1.annotate(self.names[n], xy=(x, y), xytext=(-20,5),textcoords='offset points', ha='center', va='bottom',\
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),\
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', \
                                color='red')
                )
            ax1.scatter(x,y)
            ax2.scatter(y,z)
            ax3.scatter(x,z)

            xs=node_p[0,::res]
            ys=node_p[1,::res]
            zs=node_p[2,::res]
            ax1.plot(xs,ys, alpha=0.6)
            ax2.plot(ys,zs, alpha=0.6)
            ax3.plot(xs,zs, alpha=0.6)
            ax4.plot(
                np.abs(np.linalg.norm(node_p[:,:], axis=0)-np.linalg.norm(node_p[:,0])),
                label=self.names[n],
                alpha=0.6
            )


        ax1.set_title("X-Y (Top)")
        ax2.set_title("Y-Z (Side)")
        ax3.set_title("X-Z (Front)")
        ax4.set_title("Distance from initial position (m)")
        ax4.legend(loc='upper center', ncol=3)

        return f

    def get_global_packet_logs(self, pkt_type='rx', sort=None):
        """
        Returns the global packet log, defaults to recieved packets

        Can select to be sorted by time_stamp.
        :param pkt_type: string (rx/tx)
        :param sorted: None, or Bool
        :return DataFrame:
        """
        valid_types = ['rx','tx']
        if pkt_type not in valid_types:
            raise ValueError("No such packet classification {}, valid choices are {}".format(pkt_type, valid_types))

        packets=[]
        _ =[ packets.extend(nlog[pkt_type]) for nlog in self.comms.tolist()['logs'].values()]
        if sorted is not None and sorted:
            rx_packet=sorted(packets, key=lambda k: k['time_stamp'])
        pf=pd.DataFrame(packets)
        return pf

    def get_global_trust_logs(self):
        """
        Returns the global trust log as a dict of each nodes observations at each time of each other node

        i.e. trust is internally recorded by each node wrt each node [node][t]
        for god processing it's easier to deal with [t][node]

        :return: trust observations[observer][t][target]
        """
        obs={}
        trust = { node: log['trust'] for node, log in self.comms.tolist()['logs'].items()}
        for i_node, i_t in trust.items():
            # first pass to invert the observations
            if not obs.has_key(i_node):
                obs[i_node]=[]
            for j_node, j_t in i_t.items():
                for o, observation in enumerate(j_t):
                    while len(obs[i_node])<=(o):
                        obs[i_node].append({})
                    obs[i_node][o][j_node]=observation
        return obs

    def has_comms_data(self):
        """
        Bool check if this DataPackage has useful Comms data

        :return bool:
        """

        return all([v.has_key('tx') and v.has_key('rx') for v in self.comms.tolist()['logs'].values()])

    def has_trust_data(self):
        """
        Bool check if this DataPackage has useful Trust data

        Not guaranteed for any edge cases of 'useful' such as incomplete trust data or not all nodes being 'trusty'
        :return bool:
        """

        return self.has_comms_data() and all([v.has_key('trust') for v in self.comms.tolist()['logs'].values()])
