#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:40:22 2019

@author: rob
"""

        self.params = params
        self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        self.time = 0
        self.number_of_iterations = params['batch_iterations']
        self.base_model = Model(params)
        self.models = list([deepcopy(self.base_model) for _ in range(self.number_of_particles)])  
        self.dimensions = len(self.base_model.agents2state())
        self.states = np.zeros((self.number_of_particles, self.dimensions))
        self.weights = np.ones(self.number_of_particles)
        self.indexes = np.zeros(self.number_of_particles, 'i')
        self.UKF_dict = {}
        self.AM = np.zeros((params["pop_total"],params["batch_iterations"]))
        self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        