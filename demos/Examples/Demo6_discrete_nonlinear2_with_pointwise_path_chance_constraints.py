"""
Demo 6: Tutorial 6
Type: discrete models with pointwise path chance constraints
Models:...\case_studies\discrete_time\nonlinear2.py

Description: This files shows how to solve a model discrimination problem where
it is calculated the optimal next conditions of an experiment to discriminate 3 
different models. The models are described in ...\case_studies\discrete_time\nonlinear2.py
It does not employ surrogate models. The models are assumed discrete, i.e, the next 
prediction is based on the previous state value and a change funtion. Thus, 
x1 = x0 + f(x0,u,p).
This demo has pointwise path chance constraints, mean value path constraints,
input constraints and input change constraints.
Tha pointwise path chance constraints calculate the probability of all constraints
at each point has to be respeceted, i.e, the probability that the constraint at a point
is respected "X" % of the time, where "X" is the confidence chosen by the user.
This is similar to demo design_of_experiments_for_model_discrimination_tutorial, 
however surrogate models are not used and pointwise path chance constraints are employed, 
instead of mean value constraints.

@authors: Eduardo Schultz
@version: 0
@first release: 12.06.2019
@latest release: 12.06.2019
"""

# Standard math package
import numpy as np

# Number of digits
np.set_printoptions(4)

# Package for plotting results
import matplotlib.pyplot as plt

# Number of time steps each experiment should last
num_time_steps = 50

# Number of initial experiments (before we begin optimal design of experiments)
num_init_exp = 3

# Model import
from doepy.case_studies.discrete_time import nonlinear2
from doepy.models import CandidateWrapper
from doepy.models.discrete_time import dtModel

# Model Structure Data generation
DM, Ms  = nonlinear2.get()
datagen = dtModel(CandidateWrapper(**DM.get_candidate_dict()))

# Generate experiments
U_data = DM.get_initial_experiments(num_init_exp, num_time_steps)
# Run experiments and collect measurements
Y_data = [ datagen.sample(None, U)[1] for U in U_data ]
        
# Time step sequence
T = np.arange(num_time_steps)

# Plot initial data - related to the first "num_init_exp" experiments
fig, axs = plt.subplots(1,2,figsize=(10,3))
for Y in Y_data:
    axs[0].plot(T, Y[:,0])
    axs[1].plot(T, Y[:,1])
    
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('Y_0')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('Y_1')
plt.show()


# Type of the model = Nonlinear model
from doepy.models.discrete_time import dtModel

dicts   = [ M.get_candidate_dict() for M in Ms ]

# All models (M1, M2 and M3) are now in the variable "models"
models  = [ dtModel(CandidateWrapper(**d)) for d in dicts ]


# %% Parameter estimation - Initial parameters of the models are not good
# Then the first step is to better estimate the parameters based on the 
# "num_init_exp" initial experiment data
# If the parameters are good, this part could be replaced by: 
# for i,model in enumerate(models):
#    model.p_mean = np.array(p_init)
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
    
# %% Model parameter covariance    
from doepy.param_covar import state_space_laplace

for model in models:
    model.p_covar = state_space_laplace(model, U_data)
    
# %% Objective Function
from doepy.model_discrimination.design_criteria import JR

divergence_criterion = JR()

# %% Constraints

from doepy.constraints import ControlDeltaConstraint, PointwiseChanceStateConstraint, ConstantMeanStateConstraint

# Single path chance constraint
# Confidence: 90%
d_conf = 0.90
bds = Ms[0].z_bounds
z_const = []
z_const.append(PointwiseChanceStateConstraint(bds,conf=d_conf))
z_const.append(ConstantMeanStateConstraint(bds))
#z_const = ConstantMeanStateConstraint(bds)

# Control input bounds
u_bounds = Ms[0].u_bounds

# Control change constraint
u_const  = ControlDeltaConstraint(Ms[0].u_delta)

# %% Optimisation
from doepy import ProblemInstance

PI = ProblemInstance(models, num_time_steps, divergence_criterion, u_bounds, u_const, z_const)

# The input of the first experiment is employed, but any initial guess can be used
# To avoid undesired behavior of the optimizer, it is better to use an initial guess 
# that provides an initial feasible solution
U_initial_guess = U_data[1]

# import the optimizer and the logfunction
from doepy.optimize import slsqp
from doepy.optimize import LogCallback

log   = LogCallback()
# Default option shows the results of each iteration of the optimizer
# It is possible to hide these results by setting debug=False
# Default maximum number of iterarions = 30. To change set maxmajiter=50 for example.
res   = slsqp(PI, U_initial_guess, log_callback=log, debug=True)
U_opt = res['u']
print( res['message'] )



# Load log data
data = log.load()

#
fig,axs = plt.subplots(1,2,figsize=(10,4))

N = np.arange(data['n'])  # Number of iterations
f = data['f']             # Objective function
c = np.vstack(data['c'])  # Constraints


# Plot of the results
# Note that the constraint (mean value) is respected for the different models
# however it may not be respected by the system, since the true system model (black line) 
# is unknown
# The plot uses two standart deviations (95% confidence region). 

# First plot - Optimizer results
# Left: objective function
# Right: Constraints
axs[0].set_ylabel('Obj. Function', rotation=90, fontsize=16)
axs[1].set_ylabel('Constraints', rotation=90, fontsize=16)
axs[0].set_xlabel('Iteration', fontsize=16)
axs[1].set_xlabel('Iteration', fontsize=16)
axs[0].plot(N,f) 
axs[1].plot(N,c,'k:',alpha=0.1)
plt.show()

num_exp = num_init_exp + 1

fig, axs = plt.subplots(num_exp, 2, figsize=(10, num_exp*3))
cols, ls = ['r', 'b', 'g'], ['--', '-.', ':']

num_models = 3   # Number of rival models
num_meas   = 2   # Number of measured states

Y_new = datagen.sample(None, U_opt)[1]

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
            axs[j][d].plot(T, np.ones(len(T))*bds[d][1], cols[0])
            
    # Save results
    Y_pred.append( Y )
    S_Y_pred.append( S_Y )

plt.show()