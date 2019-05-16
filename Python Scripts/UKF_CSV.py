#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:16:02 2019

@author: rob
"""

import os
os.chdir("/home/rob/DUST-RC/Python Scripts")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist


if __name__ == "__main__":
    pop_total = 300
    j = 1
    a = np.load(f"UKF_TRACKS_{pop_total}_{j}.npy")
    b =  np.load(f"ACTUAL_TRACKS_{pop_total}_{j}.npy")
    
    plt.figure()
    for j in range(int(a.shape[1]/2)):
        plt.plot(a[:,2*j],a[:,(2*j)+1])    
    
    plt.figure()
    for j in range(int(a.shape[1]/2)):
        plt.plot(b[:,2*j],b[:,(2*j)+1])    

        
    errors = True
    if errors:
        
        "find mean error between agent and prediction"
        c = {}
        c_means = []
        
        
        for i in range(int(b.shape[1]/2)):
            a_2 =   a[:,(2*i):(2*i)+2] 
            b_2 =   b[:,(2*i):(2*i)+2] 
    

            c[i] = []
            for k in range(a_2.shape[0]):
                if np.isnan(b_2[k,0]) or np.isnan(b_2[k,1]):
                    c[i].append(np.nan)
                else:                       
                    c[i].append(dist.euclidean(a_2[k,:],b_2[k,:]))
                
            c_means.append(np.nanmean(c[i]))
        
        c = np.vstack(c.values())
        time_means = np.nanmean(c,axis=0)
        plt.figure()
        plt.plot(time_means)
            
    
            
            
        index = np.where(c_means == np.nanmax(c_means))[0][0]
        print(index)
        a1 = a[:,(2*index):(2*index)+2]
        b1 = b[:,(2*index):(2*index)+2]
        plt.figure()
        plt.plot(b1[:,0],b1[:,1],label= "True Path")
        plt.plot(a1[:,0],a1[:,1],label = "KF Prediction")
        plt.legend()
    
    

    
    
    
    
    
    
    

        
  
    
