

class Behaviour():
    """
    Generic Represnetation of a Nodes behavioural characteristics
    #TODO should be a state machine?
    """
    def __init__(self,node,config):
        #TODO internal representation of the environment
        self.node=node
        self.config=config
        self.map=node.map
        self.memory={}
        self._init_behaviour()

    class memory_entry():
        def __init__(self,object_id,position):
            self.object_id=object_id
            self.position=position
            self.time=Sim.now()

    class map_entry():
        def __init__(self,node):
            self.position=node.position
            self.velocity=node.velocity
            self.time=Sim.now()

    def _init_behaviour():
        pass

    def update():
        self.map[self.node.id]=self.map_entry(self.node)
        pass

    def move():
        pass

    def addMemory(object_id,position):
        """
        Called by node lifecycle to update the internal representation of the environment
        """
        #TODO expand this to do SLAM?
        self.memory+=memory_entry(object_id,position)

    def responseVector():
        """
        Returns a 6-force-vector indicating the direction / orientation in which to move
        """
        forceVector= numpy.array([0,0,0])
        return VectorConfiguration()

    def distance(self,my_position, their_position):
        return scipy.spatial.distance.euclidean(my_position,their_position)

class Flock(Behaviour):
    """
    Flocking Behaviour as modelled by three rules:
        Short Range Repulsion
        Local-Average heading
        Long Range Attraction
    """
    def _init_behaviour(self):
        self.nearest_neighbours = config.nearest_neighbours
        self.neighbourhood_max_rad = config.neighbourhood_max_rad
        self.neighbourhood_max_dt= config.neighbourhood_max_dt
        self.neighbour_min_rad = config.neighbour_min_rad

    def _get_neighbours(self,position):
        """
        Returns an array of our nearest neighbours satisfying  the behaviour constraints set in _init_behaviour()
        """
        #Sort and filter Neighbours by distance
        neighours=filter(lambda x:x[0]<=self.neighbourhood_max_rad
        ,sorted(
            map(
                None,
                map(
                    lambda x: self.distance(position,x.position),
                    self.map
                    )
                ,self.map
                )
            ,key=itemgetter(0)
            )
        )
        #Select N neighbours in order
        return neighbours[:self.nearest_neighbours]

    def responseVector(self,position,velocity):
        """
        Called on process: Returns desired vector
        """
        forceVector= numpy.array([0,0,0])
        forceVector+= self.clumpingVector(position)
        forceVector+= self.replusiveVector(position)
        forceVector+= self.localHeading(position)

        return forceVector

    def clumpingVector(self,position):
        """
        Represents the Long Range Attraction factor:
            Head towards average fleet point
        """
        vector=numpy.array([0,0,0])
        for neighbour in neighbours:
            vector+=numpy.array(neighbour.position)

        #This assumes that the map contains one entry for each non-self node
        neighbourhood_com=vector/min(self.nearest_neighbours,len(self.map))

        # Return the fudged, relative vector to the centre of the cluster
        return (neighbourhood_com-position)/self.config.clumping_factor

    def replusiveVector(self,position):
        """
        Repesents the Short Range Repulsion behaviour:
            If a node is too close, steer away from it
        """
        #TODO Test if this is better as a scalar function rather than a step value

        vector=numpy.array([0,0,0])
        for neighbour in self._get_neighbours(position):
            if distance(position,neighbour.position) > self.neighbour_min_rad:
                #Too Close, Move away
                vector-=(position-neighbour.position)

        # Return an inverse vector to the obstacles
        return vector

    def localHeading(self,velocity):
        """
        Represents Local Average Heading
        """
        vector=numpy.array([0,0,0])
        for neighbour in self._get_neighbours(position):
            vector += neighbour.p_velocity
        return vector

    def _percieved_vector(self,node_id):
        """
        Finite Difference Estimation
        from http://cim.mcgill.ca/~haptic/pub/FS-VH-CSC-TCST-00.pdf
        """
        node_history=sorted(filter(lambda x: x.object_id==nodeid, self.memory), key=time)
        return (node_history[-1].position-node_history[-2].position)/(node_history[-1].time-node_history[-2].time)

