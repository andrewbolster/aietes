#!/usr/bin/env python
# coding=utf-8
__author__ = 'andrewbolster'
from polybos import ExperimentManager as ExpMan


def set_exp():
    e = ExpMan(node_count=4,
               title="FleetLawnmowerDriftNoiseVar",
               parallel=True, future=True,
               retain_data='files')
    e.update_default_node({
        'behaviour': 'FleetLawnmower',
        'waypoint_style': 'lawnmower',
        'positioning': 'surface',
        'drifting': 'DriftFactor',
        'ecea': 'Simple2',
        'beacon_rate': 15
    })

    default_noises = {
        'gyro_white': 0.00025,
        'gyro_color': 0.0035,
        'dvl_white': 0.004,
        'dvl_along': 0.0041,
        'dvl_across': 0.001
    }
    zeroed_noises = {
        'gyro_white': 0.0,
        'gyro_color': 0.0,
        'dvl_white': 0.0,
        'dvl_along': 0.0,
        'dvl_across': 0.0,
    }
    no_gyro = {
        'gyro_white': 0.0,
        'gyro_color': 0.0,
    }
    no_dvl = {
        'dvl_white': 0.0,
        'dvl_along': 0.0,
        'dvl_across': 0.0,
    }

    e.add_custom_node_scenario('drift_noises',
                                  [zeroed_noises, no_dvl, no_gyro, default_noises],
                                  ["No Noise", "No DVL Noise", "No Gyro Noise", "Standard Noises"]
                                  )
    e.update_duration(21600)  # 6hrs
    return e


def run_exp(e):
    e.run(title="ECEA_DriftNoise",
          runcount=32,
          )
    return e


if __name__ == "__main__":
    from aietes.Tools import memory

    exp = set_exp()
    exp = run_exp(exp)
    print exp.dump_dataruns()
    print memory()
