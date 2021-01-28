#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:49:27 2019

Experiment 0 module.

The idea here is to find a suitable range of parameters under which the UKF performs well.
We vary two parameters, namely the observation noise and sampling_rate. 

Noise provides the standard deviation to an additive Gaussian noise for our observations. 
A larger noise value gives noisy position data that is less accurate and worsens the 
quality of the observations. 0 noise gives the ground truth value. 

The sampling rate (aka assimilation rate) determines how often we assimilate forecasted
values from the ukf with our observations. A larger sampling rate allows forecasts
to run longer before being assimilated and potentially drift further from the truth.
Increasing this value significantly worsens the quality of stationsim prediction.
We require a natural number for this rate 1, 2, 3, ...

@author: rob
"""

import sys
import numpy as np

sys.path.append("../..")
# import required modules
from modules.ukf_fx import fx
from modules.ukf_plots import L2s
import modules.default_ukf_configs as default_ukf_configs

sys.path.append("../../..")
sys.path.append("../../../..")
from stationsim.ukf2 import *
from stationsim.stationsim_model import Model


def benchmark_params(n, noise, sample_rate, model_params, ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 1
    
    - assign population size, observation noise, and sampling/assimilation rate
    - assign initial covariance p as well as sensor and process noise (q,r)
    - assign transition and measurement functions (fx,hx)
    - assign observation key function and numpy file name for saving later.
    
    
    Parameters
    ------
    n, noise, sample_rate : float
        `n` population, additive `noise`, and `sample_rate` sampling rate
        
    model_params, ukf_params : dict
        dictionaries of model `model_params` and ukf `ukf_params` parameters 
        
    Returns
    ------
    model_params, ukf_params : dict
        updated dictionaries of model `model_params` and ukf `ukf_params`
        parameters ready to use in ukf_ss
    """
    # model params and init model
    model_params["pop_total"] = n
    model_params["station"] = None
    ukf_params["noise"] = noise
    ukf_params["sample_rate"] = sample_rate
    
    base_model = Model(**model_params)

    # noise structures
    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n)
    ukf_params["r"] = np.eye(2 * n)#sensor noise
    
    #fx kwargs and update function
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"] = {"base_model" : base_model}
    ukf_params["fx_kwargs_update"] = None

    #hx kwargs and update function    
    ukf_params["hx"] = hx0    
    ukf_params["hx_kwargs"] = {}
    ukf_params["obs_key_func"] = obs_key_0
    
    ukf_params["record"] = True
    ukf_params["file_name"] = f"config_agents_{n}_rate_{sample_rate}_noise_{noise}"
    
    return model_params, ukf_params, base_model

def hx0(state, **hx_kwargs):
    """ identity transition function does nothing for hx. two states are the same.
    
    Parameters
    ------
    state : array_like
        `state` vector of agent positions
    
    model_params, ukf_params : dict
        dictionaries of model `model_params` and ukf `ukf_params` parameters
    """
    return state

def obs_key_0(state, **hx_params):
    """identity function returns all agents as fully observed.
    """
    return 2 * np.ones(int(state.shape[0]/2))

def ex0_save(instance,source,f_name):
    """save grand median L2s between truths and obs,preds, and ukf
    
    - extract truths, obs, ukf predictions (preds), and forecasts
    - remove inactive agent measurements to prevent bias
    - calculate L2 distances between truth vs obs,preds and ukfs. 
    - take grand median of L2s for each estimator
    - store scalar grand medians in 3x1 numpy array (obs,forecasts,ukf)
    - save for depickle later
    
    Parameters
    -------
    
    instance : class
        ukf_ss class `instance` of finished ABM run for saving
        
    source, f_name : str
        file `source` and name `f_name`
    """
    # load in arrays of true data, observations, predictions, and assimilations
    truths = truth_parser(instance)
    nan_array= nan_array_parser(instance, truths, instance.base_model)
    obs, obs_key = obs_parser(instance, True)
    preds = preds_parser(instance, True)
    forecasts =  forecasts_parser(instance, True)
    
    # remove values for each agent after it leaves stationsim.
    # otherwise it skews the model error.
    obs *= nan_array
    truths *= nan_array
    preds *= nan_array
    forecasts *= nan_array
    
    #subset of data where assimilations happen
    truths = truths[::instance.sample_rate,:]
    preds = preds[::instance.sample_rate,:]
    forecasts = forecasts[::instance.sample_rate,:]
    obs = obs[::instance.sample_rate,:]
    
    #grand median error between truths and three estimates
    obs_error = np.nanmedian(np.nanmedian(L2s(truths,obs),axis=0))
    forecast_error = np.nanmedian(np.nanmedian(L2s(truths,forecasts),axis=0))
    ukf_error =  np.nanmedian(np.nanmedian(L2s(truths,preds),axis=0))
    
    # save
    mean_array = np.array([obs_error, forecast_error, ukf_error])
    print ("obs", "forecast", "ukf")
    print(mean_array)
    f_name = source + f_name
    print(f"saving to {f_name}")
    np.save(f_name, mean_array)
    
def ex0_main(n, noise, sampling_rate):
    """ main ex0 function 
    
    - assign population size, observation noise, and sampling rate
    - build ukf and model parameter dictionariues
    - build base model and init ukf_ss class
    - run stationsim with ukf filtering
    
    Parameters
    --------
    n, noise, sampling_rate: float
        population `n` observation `noise` and `sampling_rate`
        
    Returns
    ------
    u : cls
        `u` finished ukf instance.
    """
    # load model params
    model_params = default_ukf_configs.model_params
    # truncated model run for speed
    #model_params["step_limit"] = 100
    # load ukf params
    ukf_params = default_ukf_configs.ukf_params
    
    #update parameters for experiment 0
    model_params, ukf_params, base_model = benchmark_params(n, 
                                                            noise, 
                                                            sampling_rate, 
                                                            model_params,
                                                            ukf_params)
    print(model_params)
    print(ukf_params)
    #initiate and run ukf
    u = ukf_ss(model_params,ukf_params,base_model)
    u.main()
    # save results array via np.save
    ex0_save(u, "../../results/", ukf_params["file_name"])
    return u
  
#%%  
    
if __name__ == "__main__":
    n= 100
    noise = 1
    sample_rate = 25   
    u = ex0_main(n,  noise, sample_rate)
    
    
    
    