#!/usr/bin/env python
# coding=utf-8
"""
 * This file is part of the Aietes Framework (https://github.co/andrewbolster/aietes)
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

import uuid

from Layercake import Layercake
import Behaviour
import Applications
from aietes.Tools import *

FULL = 0
SIMPLE = 1


class Node(Sim.Process):
    """
    Generic Representation of a network node
    """

    def __init__(self, name, simulation, node_config, vector=None, **kwargs):
        self.debug = DEBUG
        self.on_mission = True

        self.id = uuid.uuid4()  # Hopefully unique id
        Sim.Process.__init__(self, name=name)
        self.logger = kwargs.get(
            "logger", simulation.logger.getChild("{0!s}[{1!s}]".format(__name__, self.name)))

        # fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
        # fileHandler.setFormatter(log_fmt)
        # self.logger.addHandler(fileHandler)

        self.simulation = simulation
        self.config = node_config
        self.mass = 37  # kg modeling remus 100

        self.settling_time = 0

        # Positions Initialised to None to highlight mistakes; as Any position
        # could be a bad position
        self.pos_log = np.empty(
            (3, self.simulation.config.Simulation.sim_duration))
        self.pos_log.fill(None)

        # Vectors initialised to Zero as at least a Zero vector doesn't break
        # things.
        self.vec_log = np.zeros(
            (3, self.simulation.config.Simulation.sim_duration))
        # However, it does make it easier to DEBUG the behave/move
        # interplay....
        self.vec_log.fill(None)

        # Store contributions
        self.contributions_log = []

        # Store any achievements
        self.achievements_log = [
            [] for _ in range(self.simulation.config.Simulation.sim_duration)]

        # Fleet Partitioning
        self.fleet = None

        # Optional Features
        self.ecea = False  # Kalman filtering of INS position
        # Drift simulation using DVR / GYR noise        #
        self.drifting = False

        # Physical Configuration
        #
        # Extract (X,Y,Z) vector from 6-vector as position
        assert len(vector) == 3, "Malformed Vector{0!s}".format(vector)
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
            application = self.config['Application']['protocol']
            if isinstance(application, list):
                if len(application) > 1:
                    raise ConfigError("Have a multi-value list where it shouldn't be! {0}".format(application))
                else:
                    application = application[0]
            app_mod = getattr(Applications, application)
        except AttributeError:
            raise ConfigError("Can't find Application: {0!s}".format(
                              node_config['app']))
        if app_mod.HAS_LAYERCAKE:
            self.layercake = Layercake(self, node_config)
        else:
            self.logger.info("No Layercake in this application")
            self.layercake = None
        self.app = app_mod(
            self, node_config['Application'], layercake=self.layercake)

        #
        # Propulsion Capabilities
        #
        if len(self.config['cruising_speed']) == 1:
            # cruising speed is independent of direction
            self.cruising_speed = np.asarray(
                [self.config['cruising_speed'][0], self.config[
                    'cruising_speed'][0], self.config['cruising_speed'][0]],
                dtype=np.float64)
        else:
            self.cruising_speed = np.asarray(
                self.config['cruising_speed'], dtype=np.float64)
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
            assert issubclass(behave_mod, Behaviour.Behaviour), behave_mod
        except AttributeError:
            raise ConfigError("Can't find Behaviour: {0}, available behaviours are ({1}) ".format(
                behaviour, map(lambda c: c.__name__, itersubclasses(Behaviour.Behaviour)))
            )

        self.behaviour = behave_mod(node=self,
                                    bev_config=self.config['Behaviour'],
                                    map=self.simulation.environment.map)  # TODO FIX SLAM MAP
        if hasattr(self.behaviour, "wallCheckDisabled"):
            self.wallCheckDisabled = self.behaviour.wallCheckDisabled
        else:
            self.wallCheckDisabled = False
        #
        # Simulation Configuration
        self.internalEvent = Sim.SimEvent(self.name)

        #
        # Kalman Filter Characteristics from Contribs
        if self.config['ecea'] == "KF":
            from contrib.Ghia.ecea.EceaFilter import ECEAFilter, ECEAParams

            confobj = ConfigObj(self.simulation.config)
            self.ecea = ECEAFilter(
                self, ECEAParams.from_aietes_conf(confobj))  #
        elif self.config['ecea'].startswith("Simple"):
            from contrib.Ghia.ecea.LSEFilter import SimpleFilter

            params_dict = {'Delta': int(self.config['beacon_rate'])}
            iterations = self.config['ecea'].split("Simple")
            if len(iterations) > 1 and iterations[1] != '':
                params_dict.update({'iterations': int(iterations[1])})
            self.tof_type = self.config['tof_type']
            if params_dict:
                self.ecea = SimpleFilter(self, params=params_dict)
            else:
                self.ecea = SimpleFilter(self)

        # Drift Characteristics from Contribs?
        if self.config['drift'] != "Null":
            import contrib.Ghia.uuv_position_drift_model.DriftFactors as Driftmodels

            drift_scales = getattr(self.config, 'drift_scales', {})
            drift_noises = getattr(self.config, 'drift_noises', {})

            self.drift = getattr(Driftmodels, self.config['drift'])(
                self, scales=drift_scales, noises=drift_noises)
            self.logger.debug("Drift activated from config: {0} with {1}, {2}".format(
                self.config['drift'], drift_scales, drift_noises))
            self.drifting = True
        elif 'drift' in kwargs:
            self.drift = kwargs.get('drift')(self)
            self.logger.debug(
                "Drift activated from kwarg: {0}".format(self.drift.__name__))
            self.drifting = True

        self.nodenum = None

    def assign_fleet(self, fleet):
        """
        Assign or Re-assign a node to a given Fleet object
        :param fleet:
        """
        self.fleet = fleet
        self.nodenum = self.fleet.nodenum(self)
        self.debug = self.debug and self.nodenum == 0
        self.update_fleet()

    def activate(self, launch_args=None):
        """
        Fired on Sim Start
        :param launch_args:
        """
        Sim.activate(self, self.lifecycle())
        if launch_args is None:
            launch_args = {}
        self.app.activate()

        # Messy nasty way to deal with some behaviours that need activating
        if hasattr(self.behaviour, 'activate'):
            self.behaviour.activate(**launch_args)

        self.update_environment()

    def mission_accomplished(self):
        """
        GWB
        """
        self.on_mission = False

    def wall_check(self):
        """
        Are we still in the bloody box?
        True if within bounds of the environment while we're doing wall checking (as defined by the behaviour)
        """
        return self.wallCheckDisabled or \
               all(self.position < np.asarray(self.simulation.environment.shape)) and \
               all(np.zeros(3) < self.position)

    def distance_to(self, their_position):
        """

        :param their_position:
        :return:
        """
        d = distance(self.get_pos(), their_position)
        return d

    def push(self, force_vector, contributions=None):
        """

        :param force_vector:
        :param contributions:
        """
        if Sim.now() < self.settling_time:
            force_vector = np.zeros(shape=force_vector.shape)
            contributions = {}
        assert len(force_vector) == 3 and not np.isnan(sum(force_vector)), "Out of spec vector: {0!s},{1!s}".format(
            force_vector, type(force_vector))
        self.acceleration_force = force_vector
        self.contributions_log.append(contributions)

    def cruise_control(self, velocity, prev_velocity):
        """
        Attempt to maintain cruising velocity
        :param velocity:
        :param prev_velocity:
            The resultant velocity should:
                y<x for cruise < x < max
                y<=max for x>max
            Candidates:
                y=1/exp(-cruise+x) << fucking insane...
                y=((2.1*(1/2+x)-1)/(1/2+x)) Too slow after x>cruise (also not easily variable)
                y=2.3*erfc(-(x-(1.4+0.09))/sqrt(1.4/2.3))/2 for x in [1..2]  The 0.09 is a fudge factor and I know it.
                y=unit(x)* (cruise + ((max - cruise)*1-exp(-(x-cruise))))) NEARLY but not numerically stable for arbitrary v-limits
        """
        cruisev = max(self.cruising_speed)
        maxv = max(self.max_speed)
        if cruisev < mag(velocity):
            new_v = unit(velocity) * (cruisev + ((maxv-cruisev)*(1-np.exp(-(mag(velocity)-cruisev)))))
            self.logger.debug("Cruising with {0!s} originally, M{1:f}, now taking {2!s}, M{3!s}".format(velocity, mag(velocity), new_v, mag(new_v)))
        else:
            new_v = velocity
        if DEBUG:
            self.logger.error("Cruise: From {0:f} against {1:f} and vel of {2:f}".format(
                mag(velocity), cruisev, mag(new_v)))
        return new_v

    def move(self):
        """
        Update node status
        """
        #
        # Positional information
        #
        old_pos = self.get_pos()
        old_vec = self.get_vec()
        dt = self.simulation.delta_t(Sim.now(), self._lastupdate)
        # Since you're an idiot and keep forgetting if this is right or not; it is;
        # src (http://physics.stackexchange.com/questions/17049/how-does-force-relate-to-velocity)
        # TL;DR
        # F = ma = dv/dt;
        # dv/dt = F(v,t)/m
        # dv/dt->(v(t+e)-v(t))/dt;
        # v(t+e) = v(t)+(F*dt)/m
        drag_force = -(old_vec**2 * 3.0 * 0.13)/2.0 # F_d = (density * velocity^2 * drag coeff * area)/2
        new_velocity = self.velocity + ((self.acceleration_force * dt) / self.mass)
        if mag(new_velocity) > max(self.cruising_speed):
            self.velocity = self.cruise_control(new_velocity, self.velocity)
            if DEBUG:
                self.logger.debug(
                    "Normalized Velocity: {0!s}, clipped: {1!s}, drag: {2!s}".format(new_velocity, self.velocity, drag_force))
        else:
            if DEBUG:
                self.logger.debug("Velocity: {0!s}".format(new_velocity))
            self.velocity = new_velocity

        real_max = max(self.max_speed*1.01) # Dealing with floating point comparisons
        assert mag(self.velocity) <= real_max, "Breaking the speed limit: {0:f}, {1:f}".format(
            mag(self.velocity), real_max
        )

        self._lastupdate = Sim.now()

        # If we're drifting, the Node's concept of reality is NOT true
        if self.drifting:
            try:
                self.position, self.velocity, error_dict = self.drift.update(old_pos,
                                                                             self.velocity,
                                                                             old_vec,
                                                                             self._lastupdate,
                                                                             dt)
            except FloatingPointError:
                err_type, value, traceback = sys.exc_info()
                raise ValueError, ("Dt,t:{0},{1}".format(
                    dt, Sim.now()), err_type, value), traceback
        else:
            self.position += self.velocity

        self.pos_log[:, self._lastupdate] = self.position.copy()
        self.vec_log[:, self._lastupdate] = self.velocity

        if DEBUG:
            self.logger.debug("Moving by {0!s} at {1!s} * {2:f} from {3!s} to {4!s}".format(
                self.velocity, mag(self.velocity), dt, old_pos, self.position))
        if not self.wall_check():
            self.logger.critical("Moving by {0!s} at {1!s} * {2:f} from {3!s} to {4!s}".format(
                self.velocity, mag(self.velocity), dt, old_pos, self.position))
            raise RuntimeError(
                "{0} Crashed out of the environment at {1}".format(self.name, Sim.now()))

        assert not np.isnan(sum(self.pos_log[:, self._lastupdate]))

        self.highest_attained_speed = max(
            self.highest_attained_speed, mag(self.velocity))

    def update_environment(self):
        """


        """
        self.simulation.environment.update(self.id,
                                           self.get_pos(true=True),
                                           self.get_vec(true=True))
        pass

    def update_fleet(self):
        """


        """
        if self.ecea:
            # if using ECEA, need to update the fleet with the *corrected* positions
            # These corrected positions are currently only used by ECEA itself, not for
            # behaviour decisions #TODO

            self.fleet.shared_map.update(self.id,
                                         self.ecea.get_pos(),
                                         self.ecea.get_vec())
        else:
            self.fleet.shared_map.update(self.id,
                                         self.get_pos(),
                                         self.get_vec())

    def fleet_preprocesses(self):
        """
        Perform operations that require fleet state to be coalesced before informing per-node behaviours
        EG Ecea
        :return:
        """
        if self.ecea:
            if not self.ecea.activated:
                self.ecea.activate()
            if (Sim.now()) % self.ecea.params.Delta == 0:
                # If in a delta period, update the kalman filter with the requires deltas
                # need to collect:
                if Sim.now() == 0:
                    true_positions = self.fleet.node_cheat_positions()
                    drifted_positions = self.fleet.node_cheat_drift_positions()
                    drifted_deltas = np.zeros_like(true_positions)
                    est_positions = None
                    # Need to cycle the filter with the 0th update
                else:
                    # This must be the real environmental data to simulate TOF
                    # Drift is based on each nodes REPORTED position from the
                    # previous REPORTED position
                    true_positions = self.fleet.node_cheat_positions()
                    drifted_positions = self.fleet.node_cheat_drift_positions()
                    last_index = max(0, Sim.now() - self.ecea.params.Delta)
                    # this_index = Sim.now()
                    last_drifted_positions = self.fleet.node_cheat_drift_positions_at(
                        last_index)  # INS Delta since last
                    drifted_deltas = drifted_positions - last_drifted_positions
                    last_true_positions = self.fleet.node_cheat_positions_at(
                        last_index)  # True Delta since last
                    # true_deltas = true_positions-last_true_positions

                if self.ecea.type == SIMPLE:
                    error_estimates = self.fleet.node_position_errors(
                    ) * self.ecea.params.Delta
                    tof = self.fleet.time_of_flight_matrix(
                        guess_index=self.tof_type)
                    # True positions only used for statistics
                    improved_positions = self.ecea.update(time_of_flight_matrix=tof,
                                                          drifted_positions=drifted_positions,
                                                          error_estimates=error_estimates,
                                                          true_positions=true_positions,
                                                          est_deltas=drifted_deltas)  # Swap these lines commenting to enable feedback
                    # last_estimate=drifted_positions) #Or disable feedback
                    if __debug__:
                        drift_d = np.linalg.norm(
                            true_positions[self.nodenum] - drifted_positions[self.nodenum])
                        improved_d = np.linalg.norm(
                            true_positions[self.nodenum] - improved_positions[self.nodenum])
                        with np.errstate(divide='ignore', invalid='ignore'):
                            self.logger.debug("{0}:{1}/{2}({3:2.2f})/{4}({5:2.2f}):{6:2.2f}%".format(
                                self.name, true_positions[self.nodenum],
                                drifted_positions[self.nodenum], drift_d,
                                improved_positions[
                                    self.nodenum], improved_d, (1.0 - (improved_d / drift_d)) * 100.0
                            ))
                else:
                    raise RuntimeError(
                        "There's no way you should be in here...:{0}".format(self.ecea))

            else:
                # Pass for processing in between deltas
                pass
            pass

    def set_pos(self, place_vector):
        """

        :param place_vector:
        """
        assert isinstance(place_vector, np.array)
        assert len(place_vector) == 3
        self.logger.info("Vector focibly moved")
        self.position = place_vector

    # noinspection PyNoneFunctionAssignment
    def get_pos(self, true=False):
        """
        Returns the current position of the node ACCORDING TO THE NODE

        If the node is drifting, the nodes position is incorrect, and
            it's 'true' position is requested using the keyword arg
        :param true:
        """
        if self.drifting and true:
            position = self.drift.get_pos()
        else:
            position = self.position.copy()
        return position

    def get_vec(self, true=False):
        """
        Returns the current velocity of the node ACCORDING TO THE NODE

        If the node is drifting, the nodes velocity is incorrect, and
            it's 'true' velocity is requested using the keyword arg
        :param true:
        """
        if self.drifting and true:
            velocity = self.drift.get_vec()
            # FIXME For the time being assume that the control output is a fixed track and don't use the KF for  updated info
            # elif self.drifting and self.ecea:
            # If using the kalman filter, get the corrected position
        # velocity = self.ecea.get_vec()
        else:
            velocity = self.velocity.copy()
        return velocity

    def grant_achievement(self, achievement):
        """
        Record an achievement for statistics
            Achievements are indexed by time and can contain any object, in a list,
            although it is assumed to be a three-vector position and distance
        :param achievement:
        """
        if isinstance(achievement, tuple):
            self.achievements_log[self._lastupdate].append(achievement)
        else:
            raise RuntimeError(
                "Non standard Achievement passed:{0!s}".format(achievement))

    def time_since_last_achievement(self):
        """


        :return:
        """
        try:
            return [i for i, l in enumerate(self.achievements_log) if len(l[i])][-1]
        except IndexError:
            return self._lastupdate

    def mean_achievement_time(self):
        """


        :return:
        """
        try:
            wins = [
                i for i, l in enumerate(self.achievements_log) if len(l[0])]
            return sum(wins) / len(wins)
        except IndexError:
            return self._lastupdate

    def lifecycle(self):
        """
        Called to update internal awareness and motion:
            THESE CALLS ARE NOT GUARANTEED TO BE ALIGNED ACROSS NODES
        """

        # Nasty ECEA initialisation because it needs to be started AFTER the nodes
        # have exposed their positions to the fleet and BEFORE t>0

        while Sim.now() < self.simulation.duration_intervals:

            yield Sim.passivate, self
            #
            # Update Node State
            #
            try:
                self.behaviour.process()
                self.app.tick()
            except Exception:
                self.logger.error("Exception in Process")
                raise

            yield Sim.passivate, self

            #
            # Move Fleet
            #
            try:
                self.move()
            except Exception:
                self.logger.error("Exception in Move")
                raise

            #
            # Update Fleet State
            # Initial fleet is instantiated in activate; so if it's the first interval, skip this
            #
            if Sim.now() > 0:
                self.update_fleet()

            yield Sim.passivate, self
            self.fleet_preprocesses()
            #
            # Update Environment State
            # Initial environment is instantiated in activate; so if it's the first interval, skip this
            #
            if Sim.now() > 0:
                try:
                    self.update_environment()
                except Exception:
                    self.logger.error("Exception in Environment Update")
                    raise

            # Clock Tick
            yield Sim.hold, self, self.behaviour.update_rate
