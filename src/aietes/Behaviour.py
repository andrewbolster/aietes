import numpy as np
from aietes.Tools import  map_entry, baselogger, distance, fudge_normal, debug, unit, mag, listfix
from operator import attrgetter


#debug = True

class Behaviour(object):
    """
    Generic Represnetation of a Nodes behavioural characteristics
    #TODO should be a state machine?
    """

    def __init__(self, *args, **kwargs):
        #TODO internal representation of the environment
        self.node = kwargs.get('node')
        self.bev_config = kwargs.get('bev_config')
        self.map = kwargs.get('map', None)
        self._start_log(self.node)
        if debug: self.logger.debug('from bev_config: %s' % self.bev_config)
        #self.logger.setLevel(logging.DEBUG)
        self.update_rate = 1
        self.memory = {}
        self.behaviours = []
        self.simulation = self.node.simulation
        self.env_shape = np.asarray(self.simulation.environment.shape)
        self.neighbours = {}

    def _start_log(self,parent):
        self.logger = parent.logger.getChild("Behaviour:%s"%(self.__class__.__name__))
        self.logger.debug('creating instance')

    def normalize_behaviour(self, forceVector):
        return (forceVector)

    def neighbour_map(self):
        """
        Returns a filtered map pulled from the map excluding self
        """
        orig_map = dict((k, v) for k, v in self.map.items() if v.object_id != self.node.id)

        # Internal Map of the neighbourhood based on best-guess of location (circa 5m / 5ms)
        for k, v in orig_map.items():
            orig_map[k].position = fudge_normal(v.position, 5)
            orig_map[k].velocity = fudge_normal(v.velocity, 5)

        return orig_map

    def addMemory(self, object_id, position, velocity):
        """
        Called by node lifecycle to update the internal representation of the environment
        """
        #TODO expand this to do SLAM?
        self.memory += map_entry(object_id, position, velocity)


    def process(self):
        """
        Process current map and memory information and update velocities
        """
        self.neighbours = self.neighbour_map()
        self.nearest_neighbours = self.getNearestNeighbours(self.node.position,
                                                            n_neighbours = self.n_nearest_neighbours)
        contributions = {}

        for behaviour in self.behaviours:
            contributions[behaviour.__name__] = behaviour(self.node.position, self.node.velocity)

        forceVector = sum(contributions.values())

        forceVector = self.avoidWall(self.node.position, self.node.velocity, forceVector)
        if debug: self.logger.info("Response:%s" % (forceVector))
        if debug:
            total = sum(map(mag, contributions.values()))

            self.logger.info("contributions: %s of %3f" % (
            [
            "%s:%.f%%" % (func, 100 * mag(value) / total)
            for func, value in contributions.iteritems()
            ], total)
            )
        forceVector = fudge_normal(forceVector, 0.012) #TODO Da Fuck is this?!
        self.node.push(forceVector, contributions = contributions)
        return

    def getNearestNeighbours(self, position, n_neighbours = None, distance = np.inf):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        """
        #Sort and filter Neighbours by distance
        neighbours_with_distance = [map_entry(key,
                                              value.position, value.velocity,
                                              name = self.simulation.reverse_node_lookup(key).name,
                                              distance = self.node.distance_to(value.position)

        ) for key, value in self.neighbours.items()]
        #self.logger.debug("Got Distances: %s"%neighbours_with_distance)
        nearest_neighbours = sorted(neighbours_with_distance
            , key = attrgetter('distance')
        )
        #Select N neighbours in order
        #self.logger.debug("Nearest Neighbours:%s"%nearest_neighbours)
        if n_neighbours is not None:
            nearest_neighbours = nearest_neighbours[:n_neighbours]
        return nearest_neighbours


    def repulseFromPosition(self, position, repulsive_position, d_limit):
        forceVector = np.array([0, 0, 0], dtype = np.float)
        distanceVal = distance(position, repulsive_position)
        forceVector = unit(position - repulsive_position) * d_limit / float(distanceVal)
        assert distanceVal > 2, "Too close to %s" % (self.getNearestNeighbours(repulsive_position)[0])
        if debug: self.logger.debug(
            "Repulsion from %s: %s, at range of %s" % (forceVector, repulsive_position, distanceVal))
        return forceVector

    def attractToPosition(self, position, attractive_position, d_limit):
        forceVector = np.array([0, 0, 0], dtype = np.float)
        distanceVal = distance(position, attractive_position)
        forceVector = unit(attractive_position - position) * float(distanceVal) / d_limit
        if debug: self.logger.debug(
            "Attraction to %s: %s, at range of %s" % (forceVector, attractive_position, distanceVal))
        return forceVector


    def avoidWall(self, position, velocity, forceVector):
        """
        Called by responseVector to avoid walls to a distance of half min distance
        """
        response = np.zeros(shape = forceVector.shape)
        min_dist = self.neighbour_min_rad * 2
        avoid = False
        if any((np.zeros(3) + min_dist) > position):
            if debug: self.logger.debug("Too Close to the Origin-surfaces: %s" % position)
            offending_dim = position.argmin()
            avoiding_position = position.copy()
            avoiding_position[offending_dim] = float(0.0)
            avoid = True
        elif any(position > (self.env_shape - min_dist)):
            if debug: self.logger.debug("Too Close to the Upper-surfaces: %s" % position)
            offending_dim = position.argmax()
            avoiding_position = position.copy()
            avoiding_position[offending_dim] = float(self.env_shape[offending_dim])
            avoid = True
        else:
            response = forceVector

        if avoid:
            #response = 0.5 * (position-avoiding_position)
            response = self.repulseFromPosition(position, avoiding_position, 1)
            #response = (avoiding_position-position)
            self.logger.error("Wall Avoidance:%s" % (response))

        return response


class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """

    def __init__(self, *args, **kwargs):
        Behaviour.__init__(self, *args, **kwargs)
        self.n_nearest_neighbours = listfix(int, self.bev_config.nearest_neighbours)
        self.neighbourhood_max_rad = listfix(int, self.bev_config.neighbourhood_max_rad)
        self.neighbour_min_rad = listfix(int, self.bev_config.neighbourhood_min_rad)
        self.clumping_factor = listfix(float, self.bev_config.clumping_factor)
        self.repulsive_factor = listfix(float, self.bev_config.repulsive_factor)
        self.schooling_factor = listfix(float, self.bev_config.schooling_factor)
        self.collision_avoidance_d = listfix(float, self.bev_config.collision_avoidance_d)

        self.behaviours.append(self.clumpingVector)
        self.behaviours.append(self.repulsiveVector)
        self.behaviours.append(self.localHeading)

        assert self.n_nearest_neighbours > 0
        assert self.neighbourhood_max_rad > 0
        assert self.neighbour_min_rad > 0

    def clumpingVector(self, position, velocity):
        """
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector = np.array([0, 0, 0], dtype = np.float)
        for neighbour in self.nearest_neighbours:
            vector += np.array(neighbour.position)

        try:
            #This assumes that the map contains one entry for each non-self node
            self.neighbourhood_com = (vector) / len(self.nearest_neighbours)
            if debug: self.logger.debug("Cluster Centre,position,factor,neighbours: %s,%s,%s,%s" % (
            self.neighbourhood_com, vector, self.clumping_factor, len(self.nearest_neighbours)))
            # Return the fudged, relative vector to the centre of the cluster
            forceVector = self.attractToPosition(position, self.neighbourhood_com, self.collision_avoidance_d)
        except ZeroDivisionError:
            self.logger.error("Zero Division Error: Returning zero vector")
            forceVector = position - position

        if debug: self.logger.debug("Clump:%s" % (forceVector))
        return self.normalize_behaviour(forceVector) * self.clumping_factor

    def repulsiveVector(self, position, velocity):
        """
        Represents the Short Range Repulsion behaviour:
            Steer away from it based on a repulsive desire curve
        """
        forceVector = np.array([0, 0, 0], dtype = np.float)
        for neighbour in self.nearest_neighbours:
            if distance(position, neighbour.position) < self.collision_avoidance_d:
                forceVector += self.repulseFromPosition(position, neighbour.position, self.collision_avoidance_d)
        # Return an inverse vector to the obstacles
        if debug: self.logger.debug("Repulse:%s" % (forceVector))
        return self.normalize_behaviour(forceVector) * self.repulsive_factor

    def localHeading(self, position, velocity):
        """
        Represents Local Average Heading
        """
        vector = np.array([0, 0, 0])
        for neighbour in self.nearest_neighbours:
            vector += fudge_normal(unit(neighbour.velocity), max(abs(unit(neighbour.velocity))) / 3)
        forceVector = (vector / (len(self.nearest_neighbours)))
        if debug: self.logger.debug("Schooling:%s" % (forceVector))
        return self.normalize_behaviour(forceVector) * self.schooling_factor

    def _percieved_vector(self, node_id, time):
        """
        Finite Difference Estimation
        from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
        """
        node_history = sorted(filter(lambda x: x.object_id == node_id, self.memory), key = time)
        return (node_history[-1].position - node_history[-2].position) / (node_history[-1].time - node_history[-2].time)


class Waypoint(Flock):
    def __init__(self, *args, **kwargs):
        Flock.__init__(self, *args, **kwargs)
        self.waypoint_factor = listfix(float, self.bev_config.waypoint_factor)
        if not hasattr(self, str(self.bev_config.waypoint_style)):
            raise ValueError("Cannot generate using waypoint definition:%s" % self.bev_config.waypoint_style)
        else:
            generator = attrgetter(str(self.bev_config.waypoint_style))
            g = generator(self)
            if __debug__ : self.logger.info("Generating waypoints: %s" % g.__name__)
            g()
        self.behaviours.append(self.waypointVector)

    def patrolCube(self):
        """
        Generates a cubic patrol loop within the environment
        """
        shape = np.asarray(self.env_shape)
        prox = 50
        cubedef = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        )
        self.cubepatrolroute = [(shape * (((vertex - 0.5) / 3) + 0.5), prox) for vertex in cubedef]
        self.nextwaypoint = waypoint(self.cubepatrolroute)
        self.nextwaypoint.makeLoop(self.nextwaypoint)


    def waypointVector(self, position, velocity):
        forceVector = np.array([0, 0, 0], dtype = np.float)
        if self.nextwaypoint is not None:
            if distance(self.neighbourhood_com, self.nextwaypoint.position) < self.nextwaypoint.prox:
                if __debug__ : self.logger.info("Moving to Next waypoint:%s" % self.nextwaypoint.position)
                self.nextwaypoint = self.nextwaypoint.next
            forceVector = self.attractToPosition(position, self.nextwaypoint.position, self.nextwaypoint.prox / 2)
        return self.normalize_behaviour(forceVector) * self.waypoint_factor


class waypoint(object):
    def __init__(self, positions):
        """
        Defines waypoint paths:
            positions = [ [ position, proximity ], *]
        """
        self.logger = baselogger.getChild("%s" % (self.__class__.__name__))
        (self.position, self.prox) = positions[0]
        if __debug__ : self.logger.info("Waypoint: %s,%s" % (self.position, self.prox))
        if len(positions) == 1:
            if __debug__ : self.logger.info("End of Position List")
            self.next = None
        else:
            self.next = waypoint(positions[1:])

    def append(self, position):
        if self.next is None:
            self.next = waypoint([position])
        else:
            self.next.append(position)

    def insert(self, position):
        temp_waypoint = self.next
        self.next = waypoint([position])
        self.next.next = temp_waypoint

    def makeLoop(self, head):
        if self.next is None:
            self.next = head
        else:
            self.next.makeLoop(head)
