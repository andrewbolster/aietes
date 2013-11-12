"""
 * This file is part of the Aietes Contrib Package
 *
 * (C) Copyright 2013 NPL, Plextek, TOM ltd
 * All Rights Reserved
 *
 * Produced under DSTL contract DSTLX1000084112
 * Subject to DEFCON 705 Usage only

 * Based on algorithmic work by Aled Catherall (Plextek Ltd)
"""
from __future__ import division

__author__ = 'Andrew Bolster, Aled Catherall'
__license__ = "DEFCON705"
__email__ = "me@andrewbolster.info"

from numpy.random import randn, seed
from numpy import pi, sqrt, exp, mod, sin, cos, zeros
import logging

logging.basicConfig(level=logging.DEBUG)

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = 16, 12


def testDriftExecution(numloops=100):
    """
    Function to demonstrate and test the drift model of an unmanned underwater vehicle
    """
    # Establish Monte Carlo Properties (let Numpy handle this)
    seed()

    # Mission Properties for test
    Duration = 8 * 60 * 60  # 8 hr mission (in s)
    dt = 1                  # time step
    x0 = 0.0                # initial position
    y0 = 0.0                # ''
    n = Duration // dt       # n-steps
    mission_types = {"straight line": 0, "big circle": 1, "lawn mower": 2}
    mission_type = mission_types['lawn mower']

    first = True

    # MC loop iteration
    error_rms = zeros((numloops, n))
    error_sum2 = zeros(n) #TODO I'm fairly certain that zeros() is faster than this.

    gyro_inits = zeros(numloops)
    dvl_inits = zeros(numloops)
    for numloop in range(numloops):
        # Progress Monitor
        percent_done = (numloop * 100.0) / numloops
        if mod(percent_done, 10) == 0:
            logging.info("numloop:%f%%" % percent_done)

        # Inititalise loop state (reset to zero mostly)
        x_est = x0 * zeros(n)
        y_est = y0 * zeros(n)
        x_true = x0 * zeros(n)
        y_true = y0 * zeros(n)
        bearing_true = 0
        bearing_est = 0

        t = 0                           # Launch Time

        bias = None                        # Sensor Biases (Delegate Default setting to DVL/GRYO fns
        bias_along = None
        bias_across = None

        gyro_inits[
            numloop] = Gyro_scale_rand_term = randn()  # Error onto Gryo bias: constant within loop but different between loops
        dvl_inits[numloop] = DVL_scale_rand_term = randn()   # Initial Value of DVL scaling error


        # Update V/Bearing
        v_along, v_across, bearing_true = update_uuv_position(t, mission_type)

        # Kick off this run
        for nStep in range(1, n):
            t += dt

            v_along_new, v_across_new, bearing_true_new = update_uuv_position(t, mission_type)

            bearing_true_rate = (bearing_true_new - bearing_true) / dt
            bearing_true = bearing_true_new

            # Obtain Doppler Velocity Log and Gryo Parameters
            v_along_est, v_across_est, bias_along, bias_across = DVL(v_along, v_across, dt, bias_along, bias_across,
                                                                     DVL_scale_rand_term, nStep)
            bearing_est_rate, bias = GYRO(bearing_true_rate, dt, bias, Gyro_scale_rand_term, nStep)

            bearing_est += bearing_est_rate * dt

            # Update estimated UUV Position
            x_est[nStep] = x_est[nStep - 1] + v_along_est * dt * cos(bearing_est) + v_across_est * dt * sin(bearing_est)
            y_est[nStep] = y_est[nStep - 1] + v_along_est * dt * sin(bearing_est) + v_across_est * dt * cos(bearing_est)

            # Update True UUV Position
            x_true[nStep] = x_true[nStep - 1] + v_along * dt * cos(bearing_true) + v_across * dt * sin(bearing_true)
            y_true[nStep] = y_true[nStep - 1] + v_along * dt * sin(bearing_true) + v_across * dt * cos(bearing_true)

            # Record Error Marker
            error_sum2[nStep] += pow((x_est[(nStep - 1)] - x_true[nStep]), 2) + pow(
                (y_est[nStep] - y_true[(nStep - 1)]), 2)

        # Record full loop error markers for test
        error_rms = sqrt(error_sum2 / numloops)
        if False:
            plt.figure(1)
            if first:
                plt.ion()
                plt.show()
                first = False
            else:
                #plt.clf()
                pass
            """
            true_path = plt.plot(x_true, y_true, 'b')
            est_path = plt.plot(x_est, y_est, 'r')
            """
            error_line = plt.plot(error_rms, 'b')
            #error_src = plt.plot(error_sum2,'r')
            plt.ylabel("x/m")
            plt.xlabel("y/m")
            plt.title("%d/%d" % (numloop, numloops))
            plt.draw()

    return error_rms, gyro_inits, dvl_inits


def DVL(v_along_true, v_across_true, dt, bias_along, bias_across, DVL_scale_rand_term, count):
    """
    Outputs estimated velocities and biases (along and across track)
    Should only be called ONCE PER SECOND
    white noise is per-measurement not per root-hour
    """
    # Scaling factors
    DVL_scale_accuracy = 0.001
    DVL_scale_error = DVL_scale_accuracy * DVL_scale_rand_term # Add random (but constant per run) term

    # White Noise conf
    std_DVL_wn = 0.004 #Speed #TODO shouldn't this be an input?
    white_noise_along = std_DVL_wn * randn()
    white_noise_across = std_DVL_wn * randn()

    # Coloured Noise conf
    std_DVL_cn_al = 0.0041 # Along noise speed
    std_DVL_cn_ac = 0.001  # Across noise speed
    tc_DVL_cn = 1800       # Markov time constant

    if bias_across is None and bias_along is None: # Initialise bias state on first timestep of run
        bias_along = std_DVL_cn_al * randn()
        bias_across = std_DVL_cn_ac * randn()
    else:
        bias_along = Markov_process(bias_along, tc_DVL_cn, std_DVL_cn_al, dt)
        bias_across = Markov_process(bias_across, tc_DVL_cn, std_DVL_cn_ac, dt)

    # Add all errors to the true velocity to give estimated velocity
    v_along_est = v_along_true * (1 + DVL_scale_error) + bias_along + white_noise_along
    v_across_est = v_across_true * (1 + DVL_scale_error) + bias_across + white_noise_across

    return v_along_est, v_across_est, bias_along, bias_across


def GYRO(bearing_true_rate, dt, bias, Gyro_scale_rand_term, count):
    """
    Outputs estimated bearing rate and gyro bias
    Can be called as often as desired; non deterministic process
    """
    # Scaling Errors
    Gyro_scale_accuracy = 0.0001
    Gyro_scale_error = Gyro_scale_accuracy * Gyro_scale_rand_term

    # White noise characteristics
    std_gyro_wn = (0.0025 / 3600) * (pi / 180)# rad / sqrt(hr)
    white_noise_term = sqrt(pow(std_gyro_wn, 2) * dt) * randn()

    # Coloured / Biased term
    std_gyro_cn = (0.0035 / 3600) * (pi / 180)# rad / hr
    tc_gyro_cn = 3600                       # Markov Time Constant

    if bias is None:                        # Initial bias state
        bias = std_gyro_cn * randn()
    else:
        bias = Markov_process(bias, tc_gyro_cn, std_gyro_cn, dt)

    bearing_est_rate = bearing_true_rate * (1 + Gyro_scale_error) + bias + white_noise_term / dt
    return bearing_est_rate, bias


def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """

    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__

# TODO Potential for memoisation as q-terms are near-static (Same per run)
@memoize
def _partial_Markov(tc, std, dt):
    q = 2 * std * std / tc
    return exp(- dt / tc) + sqrt(0.5 * q * tc * (1 - exp(-2 * dt / tc)))


def Markov_process(bias, tc, std, dt):
    bias = bias * _partial_Markov(tc, std, dt) * randn()
    return bias


def update_uuv_position(t, mission_type):
    """
    Produces the 'next step' velocity and bearing values for a given time on a given mission profile

    Locked to 8 hr missions for lawnmower
    """
    if 0 == mission_type:                   # Straight Line
        v_along_new = 1.5
        v_across_new = 0.0
        bearing_true_new = 0.0
    elif 1 == mission_type:                 # Big Circle
        v_along_new = 1.5
        v_across_new = 0.0
        bearing_true_new = 0.002 * t
    elif 2 == mission_type:                 # Lawn Mower
        t_leg = 1500
        t_turn = 125
        v_along_new = 1.5
        v_across_new = 0.0
        if t < 750:
            bearing_true_new = 0.0
        elif t < 27600:
            tt = t - 750
            leg_no = tt // t_leg
            parity = mod(leg_no, 2)
            leg_time = mod(tt, t_leg)
            if leg_time < (t_leg - t_turn):
                bearing_true_new = 0 + pi * parity
            elif parity == 0:
                bearing_true_new = (leg_time - t_leg + t_turn) * pi / t_turn
            else:
                bearing_true_new = pi - (leg_time - t_leg + t_turn) * pi / t_turn
        else:
            if t < 27600 + t_turn:
                tt = t - 27600
                bearing_true_new = pi + 0.4 * tt * pi / t_turn
            else:
                bearing_true_new = - pi + 1.1
    else:
        raise NotImplementedError("Invalid Mission ID Used:%d" % mission_type)
    return v_along_new, v_across_new, bearing_true_new


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    error_rms, gyro, dvl = testDriftExecution(numloops=10)
    plt.ioff()
    plt.figure(1)
    plt.plot(error_rms, 'r')
    plt.ylabel("error/m")
    plt.xlabel("t/s")
    plt.show()

