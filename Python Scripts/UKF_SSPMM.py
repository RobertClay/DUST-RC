#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:09:20 2019

@author: rob
"""

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
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import datetime
import time
from tqdm import tqdm as tqdm

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
        self.time_id = 0
        self.number_of_iterations = params['batch_iterations']
        self.base_model = Model(params)
        self.UKFs = {} #dictionary of KFs for each agent
        self.UKF_histories = {}
        self.M = np.zeros((params["batch_iterations"],params["pop_total"]))
        self.MC = np.zeros((params["pop_total"],params["batch_iterations"]))
        

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
                
                self.UKF_histories[j] = []
                self.UKF_histories[j].append(agent.entrance)
                
                self.UKFs[j] =  UNKF(dim_x=2,dim_z=2,fx = self.F_x, hx=self.H_x, dt=1, points=sigmas)
                self.UKFs[j].x = agent.entrance
                self.UKFs[j].R = np.diag([1,1])
                self.UKFs[j].Q = QDWN(2,dt=1,var=1)
                
                #self.F_args = {"loc_desire":agent.loc_desire,"":,"":agent.speed}
                self.UKFs[j].predict(agent)
                z = agent.location
                self.UKFs[j].update(z)#!
                self.UKF_histories[j].append(self.UKFs[j].x)


            elif self.M[self.time_id,j] == 1:
                
                self.UKFs[j].predict(agent)
                z = agent.location
                self.UKFs[j].update(z)#!
                self.UKF_histories[j].append(self.UKFs[j].x)

                
                
    def batch(self):
        time1 = datetime.datetime.now()
        for _ in tqdm(range(self.number_of_iterations-1)):
            self.step()
            self.updates()
        time2 = datetime.datetime.now()
        
        print(time2-time1)
        return

    def plots(self):
        a = UKF.UKF_histories
        b = {}
        for k in range(model_params["pop_total"]):
            b[k] =  UKF.base_model.agents[k].history_loc
        
        
        
        
        
        
        plt.figure()
        
        
        for j in range(model_params["pop_total"]):
            a1 = np.vstack(a[j])
            plt.plot(a1[:,0],a1[:,1])
            
        
        plt.figure()
        
        for j in range(model_params["pop_total"]):
            b1 = np.vstack(b[j])
            plt.plot(b1[:,0],b1[:,1])
            
          
        errors = True
        if errors:
            plt.figure()    
                
            c = {}
            c_means = []
            for j in range(model_params["pop_total"]):
                a1 = np.vstack(a[j])
                b1 = np.vstack(b[j])
                
                c[j] = []
                for k in range(a1.shape[0]-1):
                    c[j].append(dist.euclidean(a1[k+1],b1[k]))
                    
                c_means.append(np.mean(c[j]))
                
                plt.plot(np.vstack(c[j]))
                
                
                
                
            plt.figure()
            index = np.where(c_means==np.max(c_means))[0][0]
            a1 = np.vstack(a[index])
            b1 = np.vstack(b[index])
            plt.plot(b1[:,0],b1[:,1],label= "True Path")
            plt.plot(a1[:,0],a1[:,1],label = "KF Prediction")
            plt.legend()
            
            return a,b,c_means
        
        
model_params = {
        'width': 200,
        'height': 100,
        'pop_total':100,
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
UKF.batch()
a,b,c_means = UKF.plots()

