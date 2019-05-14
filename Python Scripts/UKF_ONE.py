#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:28:35 2019

@author: rob

UKF with one state
"""


import os
os.chdir("/home/rob/DUST-RC/Python Scripts")
import numpy as np
from StationSim_KM import Model, Agent
from filterpy.kalman import MerweScaledSigmaPoints as MSSP
from filterpy.kalman import UnscentedKalmanFilter as UNKF
from filterpy.common import Q_discrete_white_noise as QDWN
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm as tqdm
import pandas as pd




    
class UKF:
    """
    individually assign a UKF wrapper to each agent 
    update each agent individually depending on whether it is currently active within the model
    whether KF updates is determined by the "activity matrix" sourced at each time step from agent.active propertysqrt2 = np.sqrt(2)

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
        self.M = np.zeros((params["batch_iterations"],params["pop_total"])) #activity matrix determines if agent active
        self.MC = np.zeros((params["pop_total"],params["batch_iterations"]))
        self.finished = 0 #number of finished agents
        self.sample_rate = self.params["sample_rate"]
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
                self.M[self.time_id,j] = 0.5 #if first time indicate with .5
        self.finished = np.sum(self.M[self.time_id,:] ==2)#finished agents counter for break in batch.
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
        reciprocal_distance = np.sqrt(2) / sum(abs(loc1 - loc2))  # lerp5: profiled at 6.41μs
        x = loc2 + speed * (loc1 - loc2) * reciprocal_distance
        return x
        
        
    def H_x(location,z):
        """
        Measurement function for agent.
        !im guessing this is just the output from base_model.step
        """
        return z
    
    
    def updates(self):
        
        """
        either updates or initialises UKF else ignores agent depending on activity status
        """
        for j in range(self.pop_total):
            agent = self.base_model.agents[j]
            if self.M[self.time_id,j] == 0.5:
                "initialise"
                sigmas = MSSP(n=2,alpha=.1,beta=.2,kappa=1)
                
                self.UKF_histories[j] = [] #initialise dictionary storing state for agent j
                self.UKF_histories[j].append(agent.entrance) #record initial state
                
                self.UKFs[j] =  UNKF(dim_x=2,dim_z=2,fx = self.F_x, hx=self.H_x, dt=1, points=sigmas) #set up sigma points for filter. this calls once for filter and is stored in variate.
                self.UKFs[j].x = agent.entrance #initial state for each agent is entrance 
                self.UKFs[j].R = np.diag([1,1])*self.params["Sensor_Noise"] #sensor noise
                self.UKFs[j].Q = QDWN(2,dt=1,var=self.params["Process_Noise"]) #discrete time white noise. standard or KFs but various other error structures may be better
                
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
            if self.finished == self.pop_total:
                break
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
            c = {}
            c_means = []
            
            
            for item in a.keys():
                a_j = []
                b_j= []
                for item2 in a[item]:
                    if type(item2)!=float:
                        #item2 = item2[1:len(item2)-1]
                        #item2 = [float(x) for x in item2.split()]
                        a_j.append(np.array(item2))
                
                for item2 in b[item]:
                    if type(item2)!=float:
                        #item2 = item2[1:len(item2)-1]
                        #item2 = [float(x) for x in item2.split()]
                        b_j.append(np.array(item2))
            
            
                a_j = np.vstack(a_j)
                b_j = np.vstack(b_j)
            
                c[item] = []
                for k in range(a_j.shape[0]-1):
                    c[item].append(dist.euclidean(a_j[k+1],b_j[k]))
                    
                c_means.append(np.mean(c[item]))
                
                
                
                
            lens = [len(l) for l in c.values()]      # only iteration
            maxlen=max(lens)
            arr = np.zeros((len(c.values()),maxlen),int)
            mask = np.arange(maxlen) < np.array(lens)[:,None] # key line
            
            
            arr[mask] = np.concatenate(list(c.values()))    # fast 1d assignment
            
            
            plt.figure()
            ct = np.mean(arr,axis=0)
            plt.plot(ct)
            plt.xlabel("time")
            plt.ylabel("Mean average error (MAE)")
                
                
                
            plt.figure()
            index = np.where(c_means==np.max(c_means))[0][0]
            a1 = np.vstack(a[index])
            b1 = np.vstack(b[index])
            plt.plot(b1[:,0],b1[:,1],label= "True Path")
            plt.plot(a1[:,0],a1[:,1],label = "KF Prediction")
            plt.legend()
            

    def save_histories(self):
        a = UKF.UKF_histories
        b = {}
        for k in range(model_params["pop_total"]):
            b[k] =  UKF.base_model.agents[k].history_loc
       
        keys = list(a.keys())
        a_df = pd.DataFrame(index = np.arange(self.time_id),columns = np.arange(self.pop_total))
        for j in range(len(keys)):
            a_df[j][:len(a[j])] = a[j]
          
            
        keys = list(a.keys())
        b_df = pd.DataFrame(index = np.arange(self.time_id),columns = np.arange(self.pop_total))
        for j in range(len(keys)):
            b_df[j][:len(b[j])] = b[j]
            
        
        a_df.to_csv("UKF_tracks.csv")
        b_df.to_csv("Actual_tracks.csv")

                
        return a_df,b_df
    
    
    
if __name__ == "__main__":
    model_params = {
                'width': 200,
                'height': 100,
                'pop_total': 700,
                'entrances': 3,
                'entrance_space': 2,
                'entrance_speed': 1,
                'exits': 2,
                'exit_space': 1,
                'speed_min': .1,
                'speed_desire_mean': 1,
                'speed_desire_std': 1,
                'separation': 4,
                'wiggle': 1,
                "Sensor_Noise": 4,
                "Process_Noise": 1,
                'batch_iterations': 10_000,
                'sample_rate': 5,
                'do_save': True,
                'do_plot': True,
                'do_ani': False
                }
    
    
    UKF = UKF(Model, model_params)
    UKF.batch()
    UKF.plots()