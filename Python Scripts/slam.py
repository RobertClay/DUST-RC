# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:35:37 2019

@author: medrclaa
"""

"""
Kalman Filter for robot SLAM algorithm 
"""
from numpy.random import seed,randn
import matplotlib.pyplot as plt
from numpy import zeros,array,arange
from numpy.linalg import norm
from math import sin,cos
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

seed(10)

pos=(0, 0)
vel=(2, 1)
count=20
dt =0.1
x_var = 1
y_var = 2
landmarks = array([[0,1],[5,7],[0,10],[10,10]])   

class SLAM_PINGS(object):
    def __init__(self, pos=(0, 0), vel=(0,0), 
                 count=0,dt =0.1,
                 x_var = 0.5, y_var = 0.5,
                 landmarks = array([0,0])):
        self.vel = vel
        self.pos = [pos[0], pos[1]]
        self.times = arange(0,count,dt)
        
        #set up placeholder matrices and IC rows

        self.positions = zeros((count+1,2)) 
        self.positions[0,:] = array([pos])# start point
        self.landmark_distances = zeros((count+1,len(landmarks)))
        
        for j in range(len(landmarks)):
            self.landmark_distances[0,(j)] = norm(self.positions[0,:]-landmarks[j,:])
            #self.landmark_distances[0,(2*j)+1] = (positions[0,:]-landmarks[j,:])[1]
                
            
            
    def write(self):
        positions = self.positions
        landmark_distances = self.landmark_distances
        
        for i in range(positions.shape[0]-1):
            positions[i+1,0] =  positions[i,0] + 1 + x_var* randn()
            positions[i+1,1] =  positions[i,1] + 1 + y_var*randn()
        
            
            for j in range(len(landmarks)):
                landmark_distances[i+1,j] = norm(landmarks[j,:]-positions[i+1,:])
                #landmark_distances[i+1,(2*j)+1] = (landmarks[j,:]-positions[i+1,:])[1]


        
        return positions,landmark_distances
    
sp = SLAM_PINGS(pos,vel,count,dt,x_var,y_var,landmarks)
positions,landmark_distances = sp.write()      

      





plt.plot(positions[:,0],positions[:,1],label = "actual")
for j in range(landmark_distances.shape[1]):
    plt.plot(landmark_distances[:,j],label = f"track {j}")
plt.legend()
        