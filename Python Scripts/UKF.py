# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:08:27 2019

Rob UKF
"""
import os
os.chdir("/home/rob/DUST-RC/Python Scripts")
import numpy as np
from StationSim import Model, Agent
import filterpy as fpy
from copy import deepcopy
from multiprocessing import Pool
import multiprocessing
from filterpy.kalman import MerweScaledSigmaPoints as MSSP
from filterpy.kalman import UnscentedKalmanFilter as UNKF
from filterpy.common import Q_discrete_white_noise as QDWN
"""
UKF trajectory
"""





    
class UKF:
    """
    individually assign a UKF wrapper to each agent 
    update each agent individually depending on whether it is currently active within the model
    whether KF updates is determined by the "activity matrix" sourced at each time step from agent.active property
    if active, update KF using standard steps
    this allows for comphrehensive histories to be stored in KF objects
    
    """
    
    def __init__(self,Model,params):
        self.params = params
        self.pop_total = self.params["pop_total"]
        #self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        self.time_id = 0
        self.step_id = 0
        self.number_of_iterations = params['batch_iterations']
        self.base_model = Model(params)
        self.dimensions = len(self.base_model.agents2state())
        self.UKFs = {} #dictionary of KFs for each agent
        self.UKF_histories = {}
        self.M = np.zeros((params["batch_iterations"],params["pop_total"]))
        self.MC = np.zeros((params["pop_total"],params["batch_iterations"]))
        self.base_model.agents
        

    def step(self):
        """
        updates agents for 1 step.
        outputs activity matrix M that determines whether agent is 
        entering model, active or neither.
        For each state wish to initialise/update/ignore UKFs respectively
        """
        self.time_id += 1

        Model.step(self.base_model) #step model

        for j in range(self.pop_total): #which agents active
            self.M[self.time_id,j] = self.base_model.agents[j].active
        self.MC = np.cumsum(self.M,axis=0) #used for which agents active for first time
        
        
        for j in range(self.pop_total):
            if self.M[self.time_id,j] == 1 and self.MC[self.time_id,j] ==1 : #which active for first time
                self.M[self.time_id,j] = 0.5
        
    
        return 

  

    def F_x(location,dt,agent):
        """
        Transition function for each agent. where it is predicted to be.
        For station sim this is essentially gradient * v *dt assuming no collisions
        with some algebra to get it into cartesian plane
        """
        
        loc1 = agent.loc_desire
        loc2 = agent.location
        speed = agent.speed_desire
        distance = np.linalg.norm(loc1 - loc2) #distance between
        x = loc2 + speed * (loc1 - loc2) / distance #
        return x
        
        
    def H_x(location,z):
        """
        Measurement function for agent.
        !im guessing this is just the output from base_model.step
        """
        return z
    
    
    def updates(self):
        
        """
        either updates or initialises UKF or ignores agent depending on active status
        """
        for j in range(self.pop_total):
            agent = self.base_model.agents[j]
            if self.M[self.time_id,j] == 0.5:
                "initialise"
                sigmas = MSSP(n=2,alpha=.1,beta=.2,kappa=1)
                UKF_histories[j] = []
                UKF_histories[j].append(np.array([0,0]))
                
                self.UKFs[j] =  UNKF(dim_x=2,dim_z=2,fx = self.F_x, hx=self.H_x, dt=1, points=sigmas)
                self.UKFs[j].x = agent.entrance
                self.UKFs[j].R = np.diag([1,1])
                self.UKFs[j].Q = QDWN(2,dt=1,var=1)
                
                #self.F_args = {"loc_desire":agent.loc_desire,"":,"":agent.speed}
                self.UKFs[j].predict(agent)
                z = agent.location
                self.UKFs[j].update(z)#!

            elif self.M[self.time_id,j] == 1:
                
                self.UKFs[j].predict(agent)
                z = agent.location
                self.UKFs[j].update(z)#!
                "update"
                
    def batch(self):
        for _ in range(self.number_of_iterations-1):
            self.step()
            self.updates()
    
        return

model_params = {
        'width': 200,
        'height': 100,
        'pop_total':2,
        'entrances': 3,
        'entrance_space': 2,
        'entrance_speed': .5,
        'exits': 2,
        'exit_space': 1,
        'speed_min': .1,
        'speed_desire_mean': 1,
        'speed_desire_std': 1,
        'separation': 2,
        'batch_iterations': 4000,
        'do_save': True,
        'do_ani': True,
        }
UKF = UKF(Model, model_params)
print(UKF.batch())
a = UKF.UKFs
b = UKF.base_model.agents[0].location