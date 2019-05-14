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
    
    a = pd.read_csv("Actual_tracks.csv",dtype = str)
    a = a.drop(a.columns[0],axis=1)
    
    
    plt.figure()
    
    for item in a.columns:
        a_j = []
        for item2 in a[item]:
            if type(item2)!=float:
                item2 = item2[1:len(item2)-1]
                item2 = [float(x) for x in item2.split()]
                a_j.append(np.array(item2))
            
            
        a_j = np.vstack(a_j)
        plt.plot(a_j[:,0],a_j[:,1])
        
        
    
    b = pd.read_csv("UKF_tracks.csv",dtype = str)
    b = b.drop(b.columns[0],axis=1)
    
    plt.figure()
    
    for item in b.columns:
        b_j = []
        for item2 in b[item]:
            if type(item2)!=float:
                item2 = item2[1:len(item2)-1]
                item2 = [float(x) for x in item2.split()]
                b_j.append(np.array(item2))
            
        b_j = np.vstack(b_j)
        plt.plot(b_j[:,0],b_j[:,1])
        
        
     
    c = {}
    c_means = []
    
    
    for item in a.columns:
        a_j = []
        b_j= []
        for item2 in a[item]:
            if type(item2)!=float:
                item2 = item2[1:len(item2)-1]
                item2 = [float(x) for x in item2.split()]
                a_j.append(np.array(item2))
        
        for item2 in b[item]:
            if type(item2)!=float:
                item2 = item2[1:len(item2)-1]
                item2 = [float(x) for x in item2.split()]
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
        
        
        
        
        
    
    
    
    
    
    
    
    

        
  
    
