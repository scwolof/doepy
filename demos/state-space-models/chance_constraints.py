# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:17:55 2019

@author: edusc_000
"""

# Standard math package
import numpy as np
import math
from scipy import special as sp
#from doepy.constraints import ConstantMeanStateConstraint
#from doepy.constraints import ConstantMeanStateConstraint2 
np.set_printoptions(4)

# Package for plotting results
import matplotlib.pyplot as plt


# Number of time steps each experiment should last
num_time_steps = 50

# Number of initial experiments (before we begin optimal design of experiments)
num_init_exp = 3

from doepy.case_studies.discrete_time import nonlinear1
from doepy.models import CandidateWrapper
from doepy.models.discrete_time import dtModel

DM, Ms  = nonlinear1.get()
datagen = dtModel(CandidateWrapper(**DM.get_candidate_dict()))

# Generate experiments
U_data = DM.get_initial_experiments(num_init_exp, num_time_steps)
# Run experiments and collect measurements
Y_data = [ datagen.sample(None, U)[1] for U in U_data ]
        
# Time step sequence
T = np.arange(num_time_steps)

# Plot initial data
fig, axs = plt.subplots(1,2,figsize=(10,3))
for Y in Y_data:
    axs[0].plot(T, Y[:,0])
    axs[1].plot(T, Y[:,1])
    
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('Y_0')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('Y_1')
plt.show()

from doepy.models.discrete_time import dtNonLinearModel

dicts   = [ M.get_candidate_dict() for M in Ms ]
models  = [ dtNonLinearModel(CandidateWrapper(**d)) for d in dicts ]


# %% Parameter estimation
from doepy.param_estim.state_space import least_squares

for i,model in enumerate(models):
    # Initial guess for model parameter values
    p_init = Ms[i].p0
    # Bounds on model parameter values
    p_bounds = Ms[i].p_bounds
    # Initial states
    X0 = [ model.x0 ] * len( U_data )
    # Parameter estimation
    model.p_mean = least_squares(model, p_init, p_bounds, X0, U_data, Y_data)['x']
    
# %% Training    
#for i,model in enumerate(models):
#    p_sample_bounds = model.p_mean[:,None] * np.array([0.95, 1.05])
#    model.train(p_bounds=p_sample_bounds)
    
# %% Model parameter covariance    
from doepy.param_covar import state_space_laplace

for model in models:
    model.p_covar = state_space_laplace(model, U_data)
    
# %% Objective Function
from doepy.model_discrimination.design_criteria import JR

divergence_criterion = JR()

# %% Constraints

from doepy.constraints import ControlDeltaConstraint, ConstantMeanStateConstraint, SingleChanceStateConstraint, JointTimeChanceStateConstraint

# Path constraints
# --> Can currently only satisfy for predictive mean 
z_const  = SingleChanceStateConstraint(Ms[0].z_bounds)
z_const = ConstantMeanStateConstraint(Ms[0].z_bounds)
bds = np.array([[-0.1, 1.2],[-0.1, 1.0]])
bds = np.array([[-0.1, 1.2],[-0.1, 1.0]])
z_const = ConstantMeanStateConstraint(bds)
z_const  = SingleChanceStateConstraint(bds,conf=0.90)
z_const  = JointTimeChanceStateConstraint(bds,conf=0.90)

# Control input bounds
u_bounds = Ms[0].u_bounds

# Control change constraint
u_const  = ControlDeltaConstraint(Ms[0].u_delta)

# %% Optimisation
from doepy import ProblemInstance

PI = ProblemInstance(models, num_time_steps, divergence_criterion, u_bounds, u_const, z_const)

from doepy.optimize import slsqp
from doepy.optimize import LogCallback

U_initial_guess = U_data[0]
log   = LogCallback()
res   = slsqp(PI, U_initial_guess, log_callback=log)
U_opt = res['u']
print( res['message'] )

fig,axs = plt.subplots(1,2,figsize=(10,4))

# Load log data
data = log.load()

N = np.arange(data['n'])  # Number of iterations
f = data['f']             # Objective function
c = np.vstack(data['c'])  # Constraints

# Plot
axs[0].plot(N,f) 
axs[1].plot(N,c,'k:',alpha=0.1)
plt.show()

num_exp = num_init_exp + 1

fig, axs = plt.subplots(num_exp, 2, figsize=(10, num_exp*3))
cols, ls = ['r', 'b', 'g'], ['--', '-.', ':']

num_models = 3   # Number of rival models
num_meas   = 2   # Number of measured states

# Combine experimental data
Y_data2 = Y_data + [Y_new,]
U_data2 = U_data + [U_opt,]

# For saving predictions
Y_pred   = [ ]
S_Y_pred = [ ]

for j in range(num_exp):
    axs[j][0].set_ylabel('Experiment %d'%(j+1), rotation=90, fontsize=16)
    axs[j][0].plot(T, Y_data2[j][:,0], 'k')
    axs[j][1].plot(T, Y_data2[j][:,1], 'k', label='Data')
    
    Y   = np.zeros(( num_models, num_time_steps, num_meas ))
    S_Y = np.zeros(( num_models, num_time_steps, num_meas, num_meas ))
    
    for i,model in enumerate(models):
        # Initial latent state mean and covariance
        x0   = model.x0
        S_x0 = model.x0_covar
        
       # Compute latent state predictions
        X, S_X = model.predict_x_dist(x0, S_x0, U_data2[j])
        
        # Translate latent state predictions to observed state predictions
        Y[i], S_Y[i] = model.predict_y_dist(X[:-1], S_X[:-1])

        for d in range(2):
            # Mean and two standard deviations
            y, s  = Y[i,:,d], 1.64*np.sqrt( S_Y[i,:,d,d] )
            axs[j][d].plot(T, y, ls[i]+cols[i], label='%s'%model.name)
            axs[j][d].fill_between(T, y+s, y-s, color=cols[i], alpha=0.2)
            axs[j][d].plot(T, np.ones(len(T))*Ms[0].z_bounds[d][0], cols[0])
            #axs[j][d].plot(T, np.ones(len(T))*Ms[0].z_bounds[d][1], cols[0])
            axs[j][d].plot(T, np.ones(len(T))*bds[d][1], cols[0])
            
    # Save results
    Y_pred.append( Y )
    S_Y_pred.append( S_Y )

plt.show()