__author__ = 'bolster'
import logging
from scipy.spatial.distance import pdist, squareform
# used to reconstitute config from NP object
from ast import literal_eval

import numpy as np

from aietes.Tools import mag, add_ndarray_to_set

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
                   'names': 'names',
                   'contributions': 'contributions',
                   'achievements': 'achievements',
                   'environment': 'environment',
                   'config': 'config',
                   'title': 'title'
    }

    def _handle_mapping_exceptions_(self, source, sink, exp):
        """
        Attempts to extract DataPackage data from a source dict
        """
        #Per Attrib Fixes in the case of failure
        if sink is "title":
            # If simulation title not explicitly given, use the filename -npz
            self.title = ""
        else:
            self.logging.error("Can't find %s in source" % source)
            raise exp

    def __init__(self, source=None, *args, **kwargs):
        """
        Raises IOError on load failure
        """
        if source is not None:
            try:
                source_dataset = np.load(source)
                for sink_attrib, source_attrib in self._attrib_map.iteritems():
                    try:
                        self.__dict__[sink_attrib] = source_dataset[source_attrib]
                    except KeyError as exp:
                        self._handle_mapping_exceptions_(sink_attrib, source_attrib, exp)
            except KeyError:
                raise
        elif all(x is not None for x in [kwargs.get(attr) for attr in self._attrib_map.keys()]):
            # kwargs needed
            for sink_attrib, source_attrib in self._attrib_map.iteritems():
                try:
                    self.__dict__[sink_attrib] = kwargs[sink_attrib]
                except AttributeError as exp:
                    self._handle_mapping_exceptions_(sink_attrib, source_attrib, exp)
        else:
            raise ValueError("Can't work out what the hell you want!: %s missing" %
                             str([attr
                                  for attr, val in [(attr, kwargs.get(attr))
                                                    for attr in self._attrib_map.keys()]
                                  if val is None
                             ]))

        self.tmax = kwargs.get("tmax", len(self.p[0][0]))
        self.n = len(self.p)

        #Fixes for Datatypes
        self.config = literal_eval(str(self.config))
        if isinstance(self.names, np.ndarray):
            self.names = self.names.tolist()


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

    def write(self, filename=None):
        #TODO TEST

        np.savez(filename if filename is not None else self.filename,
                 positions=positions,
                 vectors=vectors,
                 names=names,
                 environment=self.environment.shape,
                 contributions=contributions,
                 achievements=achievements
        )
        co = ConfigObj(self.config, list_values=False)
        co.filename = "%s.conf" % filename
        co.write()

    #Data has the format:
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

    def package_statistics(self):
        """
        Generate General Package Statistics, i.e. full simulation statistics
        """

        stats = {}
        """Achievement Statistics"""
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
        ach_stat['percent_completion'] = float(len(top_guns)) / len(ach_stat['pernode'])

        stats['achievements'] = ach_stat

        """Volume Statistics"""
        #TODO The maths for this is insane....

        """Motion Statistics"""
        mot_stat = {}
        #Standard Variation of the Internode Distance Average would represent the variability of the fleet
        mot_stat["std_of_INDA"] = np.std([self.inter_distance_average(t) for t in xrange(self.tmax)])
        #Fleet speed (i.e. total distance covered) would represent comparative efficiency
        mot_f_distance = np.sum(map(mag, self.v))
        mot_stat["fleet_distance"] = mot_f_distance
        mot_stat["fleet_efficiency"] = (mot_f_distance / self.tmax) / self.n
        mot_stat["std_of_INDD"] = np.std(
            [self.position_matrix(t) / self.inter_distance_average(t) for t in xrange(self.tmax)])

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
            logging.debug("Position Query for n:%d @ all for position shape %s" % (node, self.p[node].shape))
            raise e

    def heading_slice_of(self, node):
        """
        Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
        """
        try:
            return self.v[node, :, :]
        except IndexError as e:
            logging.debug("Heading Query for n:%d @ all for heading shape %s" % (node, self.v[node].shape))
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
            raise ValueError("Time must be in the range of the dataset: %s, %s" % (time, self.tmax))

        return np.average(self.position_slice(time), axis=0)

    def average_heading(self, time):
        """
        Generate the average heading for the fleet at a given timeslice
        :param time: time index to calculate at
        :type int

        :raises ValueError
        """
        if not (0 <= time <= self.tmax):
            raise ValueError("Time must be in the range of the dataset: %s, %s" % (time, self.tmax))

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
            logging.debug(
                "Contribution Query for n:%d @ all for position shape %s" % (node, self.contributions[node].shape))
            raise e

    def inter_distance_average(self, time):
        """
        Returns the average distance between nodes
        """
        return np.average(self.position_matrix(time))
