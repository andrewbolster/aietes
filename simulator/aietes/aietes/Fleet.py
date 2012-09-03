from aietes.Tools import Sim, baselogger, debug

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


