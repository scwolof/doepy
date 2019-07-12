# -*- coding: utf-8 -*-
"""
Demo 8: Tutorial 8
Type: continuous models without constraints
Models:...\case_studies\continuous_time\EspieMacchietto1989.py

Description:

@authors: Eduardo Schultz
@version: 0
@first release: 16.06.2019
@latest release: 16.06.2019
"""

# Standard math package
import numpy as np

# Number of digits
np.set_printoptions(4)

# Package for plotting results
import matplotlib.pyplot as plt

# Number of time steps each experiment should last
num_time_steps = 40

# Number of initial experiments (before we begin optimal design of experiments)
num_init_exp = 1

# Model import
from doepy.case_studies.continuous_time import Penicillin
from doepy.models import CandidateWrapper
from doepy.models.continuous_time import ctModel

# Model Structure Data generation
Ms  = Penicillin.get()
dicts   = [ M.get_candidate_dict() for M in Ms ]
models  = [ ctModel(CandidateWrapper(**d)) for d in dicts ]
models[0].p_mean = Ms[0].p0
# Generate experiments
rndNumber = np.random.rand( num_init_exp, num_time_steps , Ms[0].num_inputs)
#U_data = Ms[0].u_bounds[:,0] + rndNumber * (Ms[0].u_bounds[:,1] - Ms[0].u_bounds[:,0])
U_data = np.array([[[0.357393276146826], [0.189995027967076], [0.214918271897468], [0.235901569329959], [0.265413275973404], [0.292244540491240], [0.327446357715214], [0.361202062637020], [0.403355913497615], [0.445273663437278], [0.495736302621640], [0.547223701325015], [0.607682755637315], [0.670314069805787], [0.742519916470610], [0.818008178657174], [0.903812305757021], [0.993950582656643], [1.09525940478916], [1.20175154013002], [1.32073346855268], [1.44514583338448], [1.58341717709197], [1.72754168589524], [1.88675373843152], [2.05184461044603], [2.23353972976418], [2.41954348140554], [2.62772397458359], [0.735237871506638], [0.585418298252679], [0.575676547550344], [0.550127243044627], [0.530769404046184], [0.529398542455452], [0.513448384537111], [0.515851680313466], [0.516053242790607], [0.511111168492188], [0.508948548446982]]])
uopt2 = np.array([[[0.357469621598674], [0.189202254889286], [0.215403889040360], [0.235615413221123], [0.265421883465561], [0.292395497615007], [0.327087096086509], [0.361445542662720], [0.403088028823661], [0.445620141920965], [0.495289982012320], [0.547663798558397], [0.607237769897692], [0.670865230302960], [0.742094270761622], [0.818584827032105], [0.902534071726810], [0.994345288221235], [1.09361843669049], [1.20159025599504], [1.31799279843248], [1.44375584312614], [1.57868958386941], [1.72349859945503], [1.87795463634392], [2.04245423039365], [2.21681356315447], [2.40103244322214], [2.59481726006121], [0.735225258760065], [0.586223531038944], [0.576551895492295], [0.548898997221581], [0.527236455182147], [0.529830952347020], [0.517129035939704], [0.511047337923047], [0.507821273447337], [0.504868414527987], [0.500746343757396]]])
#U_data = np.array([[[0.357393276146826], [0.189995027967076], [0.214918271897468], [0.235901569329959], [0.265413275973404], [0.292244540491240], [0.327446357715214], [0.361202062637020], [0.403355913497615], [0.445273663437278]]])
N = num_time_steps
D = Ms[0].num_inputs
U = U_data.reshape(( N, D ))



#from doepy.param_covar import state_space_laplace

#for model in models:
#    model.p_covar = state_space_laplace(model, U_data[0])
models[0].p_covar = Ms[0].p0_covar

# Run experiments and collect measurements
#X_data, Y_data = models[0].predict(None, U_data[0])
#X, S_X, dox = models[0].predict_x_dist(models[0].x0, Ms[0].S_x0, U_data[0], grad=True)
        
# Translate latent state predictions to observed state predictions
#Y, S_Y = models[0].predict_y_dist(X[:-1], S_X[:-1])

        
# Time step sequence
T_x = np.arange(num_time_steps+1)
T = np.arange(num_time_steps)

y = np.zeros([6,num_time_steps])
s = np.zeros([6,num_time_steps])
'''
for d in range(4):
    y[d], s[d]  = Y[:,d], 1.64*np.sqrt( S_Y[:,d,d] )
# Plot initial data - related to the first "num_init_exp" experiments
fig, axs = plt.subplots(1,2,figsize=(10,3))

#for Y in Y_data:
axs[0].plot(T, Y_data[:,1])
axs[0].plot(T_x, X_data[:,1])
axs[0].fill_between(T, y[1]+s[1], y[1]-s[1], color='b', alpha=0.2)
axs[1].plot(T, Y_data[:,2])
axs[1].plot(T_x, X_data[:,2])
axs[1].fill_between(T, y[2]+s[2], y[2]-s[2], color='b', alpha=0.2)
    
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('Y_0')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('Y_1')
plt.show()
'''

  
# %% Objective Function
from doepy.model_discrimination.design_criteria import MaxEndPoint

divergence_criterion = MaxEndPoint()

# %% Constraints

from doepy.constraints import ControlDeltaConstraint, SingleChanceStateConstraint, ConstantMeanStateConstraint

# Single path chance constraint
# Confidence: 90%
d_conf = 0.9
bds = Ms[0].z_bounds
z_const = []
#z_const.append(ConstantMeanStateConstraint(bds))
z_const.append(SingleChanceStateConstraint(bds,conf=d_conf))
#z_const = ConstantMeanStateConstraint(bds)

# Control input bounds
u_bounds = Ms[0].u_bounds

# Control change constraint
u_const = []
#u_const  = ControlDeltaConstraint(Ms[0].u_delta)

# %% Optimisation
from doepy import dop
t0 = 0
tf = 40
num_time_steps = 40
#t_const=np.array([1,3,6])
disc1 = 41
t_const2=np.linspace(t0,tf,1001, True)
t_const=np.linspace(t0,tf,disc1, True)
#t_const = t_const2
#print(t_const)
#print(t_const)
PI = dop(models, num_time_steps, divergence_criterion, u_bounds, u_const, z_const,  t_const=t_const, t0=t0, tf=tf)

from doepy.optimize import slsqp
from doepy.optimize import LogCallback

log   = LogCallback()
res   = slsqp(PI, U_data[0], log_callback=log, debug=True)
U_opt = res['u']
print( res['message'] )


#from doepy.optimize import SIP

#OP = SIP(PI, U_data[0])
#U = np.array([[0.3303, 0.1755, 0.1974, 0.2205, 0.2457, 0.2736, 0.3047, 0.3397,
#        0.3789, 0.4226, 0.4702, 0.521 , 0.5749, 0.6323, 0.6937, 0.7597,
#        0.8306, 0.9068, 0.9884, 1.0755, 1.1682, 1.2665, 1.3703, 1.4799,
#        1.5954, 1.7171, 1.847 , 1.9868, 2.0464, 0.8267, 0.6235, 0.5427,
#        0.513 , 0.493 , 0.4786, 0.4727, 0.4672, 0.4613, 0.4578, 0.4553]])
#U = np.array([[0.3303], [0.1755], [0.1974], [0.2205], [0.2457], [0.2736], [0.3047], [0.3397],[0.3789], [0.4226]])

#res = OP()

#a,b,c,d = PI(U_data[0], grad=True)
'''
a,b = PI.simulate(U)

#c,d = PI(U_data[0], grad=False)

s  = 1.64*np.sqrt( b[:,0,1,1] )
plt.plot(PI.t_all, 0.5-a[:,0,1])
#plt.plot(t_const,cn[:,3])
plt.fill_between(PI.t_all, 0.5-a[:,0,1]+s,0.5-a[:,0,1]-s,color='b',alpha=0.2)
plt.show()


PI.update_t(t_const2)
a,b = PI.simulate(U)

#c,d = PI(U_data[0], grad=False)

s  = 1.64*np.sqrt( b[:,0,1,1] )
plt.plot(PI.t_all, 0.5-a[:,0,1])
#plt.plot(t_const,cn[:,3])
plt.fill_between(PI.t_all, 0.5-a[:,0,1]+s,0.5-a[:,0,1]-s,color='b',alpha=0.2)
plt.show()



cn = b.reshape(disc1,8)
di = cn.reshape(disc1,2*PI.num_meas)
print(cn.shape)
ind = np.unravel_index(np.argmin(cn, axis=None), cn.shape)
print(ind)
print(np.min(cn))
minc = cn[ind]
ti_min = PI.t_const[ind[0]]
t_const = np.unique(np.concatenate((t_const,ti_min),axis=None))
t_const.sort()
print(t_const)


plt.plot(PI.t_const,cn[:,3])


PI.update_t(t_const2)
a,b = PI(U_data[0], grad=False)
cn = b.reshape(400,8)
di = cn.reshape(400,2*PI.num_meas)
print(cn.shape)
ind = np.unravel_index(np.argmin(cn, axis=None), cn.shape)
print(ind)
print(np.min(cn))
minc = cn[ind]
plt.plot(PI.t_const,cn[:,3])
#plt.plot(t_const,cn[:,3])
# The input of the first experiment is employed, but any initial guess can be used
# To avoid undesired behavior of the optimizer, it is better to use an initial guess 
# that provides an initial feasible solution
U_initial_guess = U_data[0]

# import the optimizer and the logfunction
#from doepy.optimize import slsqp
#from doepy.optimize import LogCallback

#log   = LogCallback()
# Default option shows the results of each iteration of the optimizer
# It is possible to hide these results by setting debug=False
# Default maximum number of iterarions = 30. To change set maxmajiter=50 for example.
#res   = slsqp(PI, U_initial_guess, log_callback=log, debug=True)
#U_opt = res['u']
#print( res['message'] )



# Load log data
#data = log.load()

#
#fig,axs = plt.subplots(1,2,figsize=(10,4))

#N = np.arange(data['n'])  # Number of iterations
#f = data['f']             # Objective function
#c = np.vstack(data['c'])  # Constraints


# Plot of the results
# Note that the constraint (mean value) is respected for the different models
# however it may not be respected by the system, since the true system model (black line) 
# is unknown
# The plot uses two standart deviations (95% confidence region). 

# First plot - Optimizer results
# Left: objective function
# Right: Constraints
#axs[0].set_ylabel('Obj. Function', rotation=90, fontsize=16)
#axs[1].set_ylabel('Constraints', rotation=90, fontsize=16)
#axs[0].set_xlabel('Iteration', fontsize=16)
#axs[1].set_xlabel('Iteration', fontsize=16)
#axs[0].plot(N,f) 
#axs[1].plot(N,c,'k:',alpha=0.1)
#plt.show()
'''
'''
fig, axs = plt.subplots(1, 2, figsize=(10, 1))
cols, ls = ['r', 'b', 'g'], ['--', '-.', ':']

X, S_X = models[0].predict_x_dist(models[0].x0, Ms[0].S_x0, U_opt)
        
# Translate latent state predictions to observed state predictions
Y, S_Y = models[0].predict_y_dist(X[:-1], S_X[:-1])

        
# Time step sequence
T_x = np.arange(num_time_steps+1)
T = np.arange(num_time_steps)

y = np.zeros([4,num_time_steps])
s = np.zeros([4,num_time_steps])
for d in range(4):
    y[d], s[d]  = Y[:,d], 1.64*np.sqrt( S_Y[:,d,d] )
# Plot initial data - related to the first "num_init_exp" experiments
fig, axs = plt.subplots(1,2,figsize=(10,3))

#for Y in Y_data:
axs[0].plot(T, Y[:,1])
axs[0].plot(T_x, X[:,1])
axs[0].fill_between(T, y[1]+s[1], y[1]-s[1], color='b', alpha=0.2)
axs[1].plot(T, Y[:,2])
axs[1].plot(T_x, X[:,2])
axs[1].fill_between(T, y[2]+s[2], y[2]-s[2], color='b', alpha=0.2)
    
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('Y_0')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('Y_1')
plt.show()
'''