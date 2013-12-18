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

import uuid
import sys

from Layercake import Layercake
import Behaviour
import Applications
from aietes.Tools import *


class Node(Sim.Process):
    """
    Generic Representation of a network node
    """

    def __init__(self, name, simulation, node_config, vector=None, **kwargs):
        self.id = uuid.uuid4()  # Hopefully unique id
        Sim.Process.__init__(self, name=name)
        self.logger = kwargs.get("logger", simulation.logger.getChild("%s[%s]" % (__name__, self.name)))

        self.simulation = simulation
        self.config = node_config
        self.mass = 10  # kg modeling remus 100
        self.mass = 5   # fudge cus I'm bricking it #FIXME



        # Positions Initialised to None to highlight mistakes; as Any position could be a bad position
        self.pos_log = np.empty((3, self.simulation.config.Simulation.sim_duration))
        self.pos_log.fill(None)

        # Vectors initialised to Zero as at least a Zero vector doesn't break things.
        self.vec_log = np.zeros((3, self.simulation.config.Simulation.sim_duration))
        # However, it does make it easier to debug the behave/move interplay....
        self.vec_log.fill(None)

        # Store contributions
        self.contributions_log = []

        # Store any achievements
        self.achievements_log = [[] for _ in range(self.simulation.config.Simulation.sim_duration)]

        #
        # Physical Configuration
        #
        # Extract (X,Y,Z) vector from 6-vector as position
        assert len(vector) == 3, "Malformed Vector%s" % vector
        self.position = np.array(vector, dtype=np.float)
        # Implied six vector velocity
        self.velocity = np.array([0, 0, 0], dtype=np.float)
        self.acceleration_force = np.array([0, 0, 0], dtype=np.float)

        self.highest_attained_speed = 0.0

        self._lastupdate = Sim.now()

        #
        # Application and(or) Comms stack
        #
        try:
            app_mod = getattr(Applications, str(node_config['app']))
        except AttributeError:
            raise ConfigError("Can't find Application: %s" % node_config['app'])
        if app_mod.HAS_LAYERCAKE:
            self.layercake = Layercake(self, node_config)
        else:
            self.logger.info("No Layercake in this application")
            self.layercake = None
        self.app = app_mod(self, node_config['Application'], layercake=self.layercake)

        #
        # Propulsion Capabilities
        #
        if len(self.config['cruising_speed']) == 1:
            # cruising speed is independent of direction
            self.cruising_speed = np.asarray(
                [self.config['cruising_speed'][0], self.config['cruising_speed'][0], self.config['cruising_speed'][0]],
                dtype=np.float64)
        else:
            self.cruising_speed = np.asarray(self.config['cruising_speed'], dtype=np.float64)
        assert len(self.cruising_speed) == 3

        if len(self.config['max_speed']) == 1:
            # Max speed is independent of direction
            self.max_speed = np.asarray(
                [self.config['max_speed'][0], self.config['max_speed'][0], self.config['max_speed'][0]])
        else:
            self.max_speed = np.asarray(self.config['max_speed'])
        assert len(self.max_speed) == 3

        if len(self.config['max_speed']) == 1:
            # Max Turn Rate is independent of orientation
            self.max_turn = [self.config['max_turn'], self.config['max_turn'], self.config['max_turn']]
        else:
            self.max_turn = self.config['max_turn']
        assert len(self.max_turn) == 3

        #
        # Internal Configure Node Behaviour
        #
        behaviour = None
        try:
            behaviour = self.config['Behaviour']['protocol']
            behave_mod = getattr(Behaviour, str(behaviour))
        except AttributeError:
            raise ConfigError("Can't find Behaviour: %s" % behaviour)

        self.behaviour = behave_mod(node=self,
                                    bev_config=self.config['Behaviour'],
                                    map=self.simulation.environment.map) ##TODO FIX SLAM MAP

        #
        # Simulation Configuration
        self.internalEvent = Sim.SimEvent(self.name)


        #
        # Fleet Partitioning
        self.fleet = None

        #
        # Drift Characteristics from Contribs?
        self.drifting = False
        if self.config['drift'] != "Null":
            import contrib.Ghia.uuv_position_drift_model as Driftmodels

            self.drift = getattr(Driftmodels, self.config['drift'])(self)
            self.drifting = True
        elif kwargs.has_key('drift'):
            self.drift = kwargs.get('drift')(self)
            self.drifting = True

        self.logger.debug('instance created')

    def activate(self, launch_args=None):
        """
        Fired on Sim Start
        """
        self.logger.debug("Initialised Node Lifecycle")
        Sim.activate(self, self.lifecycle())
        if launch_args is None:
            launch_args = {}
        self.app.activate()
        if self.app.layercake:
            self.layercake.activate()
        self.behaviour.activate(**launch_args)

        # Tell the environment that we are here!
        self.update_environment()

    def assignFleet(self, fleet):
        """
        Assign or Re-assign a node to a given Fleet object
        """
        self.fleet = fleet

    def wallCheck(self):
        """
        Are we still in the bloody box?
        """
        return all(self.position < np.asarray(self.simulation.environment.shape)) and all(np.zeros(3) < self.position)

    def distanceTo(self, otherNode):
        assert hasattr(otherNode, "position"), "Other object has no position"
        assert len(otherNode.position) == 3
        return distance(self.position, otherNode.position)

    def push(self, forceVector, contributions=None):
        assert len(forceVector) == 3 and not np.isnan(sum(forceVector)), "Out of spec vector: %s,%s" % (
            forceVector, type(forceVector))
        self.acceleration_force = forceVector
        self.contributions_log.append(contributions)

    def cruiseControl(self, velocity, prev_velocity):
        """
        Attempt to maintain cruising velocity
            The resultant velocity should:
                y<x for cruise < x < max
                y<=max for x>max
            Candidates:
                y=1/exp(-cruise+x) << fucking insane...
                y=((2.1*(1/2+x)-1)/(1/2+x)) Too slow after x>cruise (also not easily variable)
                y=2.3*erfc(-(x-(1.4+0.09))/sqrt(1.4/2.3))/2 for x in [1..2]  The 0.09 is a fudge factor and I know it.

        """
        cruisev = max(self.cruising_speed)
        maxv = max(self.max_speed)
        if cruisev < mag(velocity) < maxv:
            new_V = unit(velocity) * (cruisev + (mag(velocity) - cruisev) ** 2)
        elif maxv < mag(velocity):
            new_V = unit(velocity) * max(self.max_speed)
        else:
            self.logger.error("shouldn't really be here: {},{}".format(velocity, mag(velocity)))
        if debug:
            self.logger.error("Cruise: From %f against %f and vel of %f" % (
                mag(velocity), cruisev, mag(new_V)))
        return new_V

    def move(self):
        """
        Update node status
        """
        #
        # Positional information
        #
        old_pos = self.position.copy()
        old_vec = self.velocity.copy()
        dT = self.simulation.deltaT(Sim.now(), self._lastupdate)
        # Since you're an idiot and keep forgetting if this is right or not; it is;
        # src (http://physics.stackexchange.com/questions/17049/how-does-force-relate-to-velocity)
        # TL;DR
        # F = ma = dv/dt;
        # dv/dt = F(v,t)/m
        # dv/dt->(v(t+e)-v(t))/dt;
        # v(t+e) = v(t)+(F*dt)/m
        new_velocity = self.velocity + ((self.acceleration_force * dT) / self.mass)
        if mag(new_velocity) > max(self.cruising_speed):
            self.velocity = self.cruiseControl(new_velocity, self.velocity)
            if debug:
                self.logger.debug("Normalized Velocity: %s, clipped: %s" % (new_velocity, self.velocity))
        else:
            if debug:
                self.logger.debug("Velocity: %s" % new_velocity)
            self.velocity = new_velocity

        assert mag(self.velocity) < max(self.max_speed), "Breaking the speed limit: %s, %s" % (
            mag(self.velocity), self.cruising_speed
        )

        if self.drifting:
            try:
                self.position, self.velocity, error_dict = self.drift.update(self.position,
                                                                             self.velocity,
                                                                             old_vec,
                                                                             self._lastupdate,
                                                                             dT)
            except FloatingPointError:
                type, value, traceback = sys.exc_info()
                raise ValueError, ("Dt,t:{},{}".format(dT, Sim.now()), type, value), traceback
        else:
            self.position += self.velocity

        if debug:
            self.logger.debug("Moving by %s at %s * %f from %s to %s" % (
                self.velocity, mag(self.velocity), dT, old_pos, self.position))
        if not self.wallCheck():
            self.logger.critical("Moving by %s at %s * %f from %s to %s" % (
                self.velocity, mag(self.velocity), dT, old_pos, self.position))
            self.logger.critical("WE'RE OUT OF THE ENVIRONMENT! %s, v=%s" % (self.position, self.velocity))
            self.logger.critical("PosLog:(%d,%d)\n%s" % (
                self._lastupdate, 0, [mag(self.pos_log[:, x]) for x in range(self._lastupdate)]))
            raise Exception("%s Crashed out of the environment at %s m/s" % (self.name, mag(self.velocity)))

        self.pos_log[:, self._lastupdate] = self.position.copy()
        self.vec_log[:, self._lastupdate] = self.velocity

        assert not np.isnan(sum(self.pos_log[:, self._lastupdate]))

        self.highest_attained_speed = max(self.highest_attained_speed, mag(self.velocity))
        self._lastupdate = Sim.now()

    def update_environment(self):
        if not self.drifting:
            self.simulation.environment.update(self.id,
                                               self.getPos(),
                                               self.getVec())
        else:
            self.simulation.environment.update(self.id,
                                               self.drift.getPos(),
                                               self.drift.getVec())

    def setPos(self, placeVector):
        assert isinstance(placeVector, np.array)
        assert len(placeVector) == 3
        self.logger.info("Vector focibly moved")
        self.position = placeVector

    def getPos(self):
        return self.position.copy()

    def getVec(self):
        return self.velocity.copy()

    def distance_to(self, their_position):
        d = distance(self.getPos(), their_position)
        return d

    def grantAchievement(self, achievement):
        """
        Record an achievement for statistics
            Achievements are indexed by time and can contain any object, in a list,
            although it is assumed to be a three-vector position and distance
        """
        if isinstance(achievement, tuple):
            self.achievements_log[self._lastupdate].append(achievement)
        else:
            raise RuntimeError("Non standard Achievement passed:%s" % achievement)

    def lifecycle(self):
        """
        Called to update internal awareness and motion:
            THESE CALLS ARE NOT GUARANTEED TO BE ALIGNED ACROSS NODES
        """
        while True:
            #
            # Update Node State
            #
            if debug:
                self.logger.info('updating behaviour')
            try:
                self.behaviour.process()
            except Exception:
                self.logger.error("Exception in Process")
                raise

            yield Sim.passivate, self

            #
            # Move Fleet
            #
            if debug:
                self.logger.info('updating position, then waiting %s' % self.behaviour.update_rate)
            try:
                self.move()
            except Exception:
                self.logger.error("Exception in Move")
                raise

            #
            # Update Fleet State
            #
            if debug:
                self.logger.info('updating map')
            yield Sim.hold, self, self.behaviour.update_rate
            try:
                self.update_environment()
            except Exception:
                self.logger.error("Exception in Environment Update")
                raise
