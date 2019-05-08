#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:56:27 2019

@author: rob
"""

import os
os.chdir("/home/rob/DUST-RC/Python Scripts")
import numpy as np
from StationSim import Model, Agent
import filterpy as fpy
from copy import deepcopy
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

model_params = {
        'width': 200,
        'height': 100,
        'pop_total':1,
        'entrances': 3,
        'entrance_space': 2,
        'entrance_speed': .1,
        'exits': 2,
        'exit_space': 1,
        'speed_min': .1,
        'speed_desire_mean': 1,
        'speed_desire_std': 1,
        'separation': 2,
        'batch_iterations': 50,
        'do_save': True,
        'do_ani': True,
        }

locs,agents = Model(model_params).batch()


def F(agent):
    agents[0].