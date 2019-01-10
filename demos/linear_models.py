
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

import gpflow.training as train

from GPdode.case_studies import cs1
from GPdode import ProblemInstance

models = cs1.get()
u_bnds = np.array([[0., 0.5]])

# Create problem instance
problem = ProblemInstance(models, 50, u_bnds)
# Optimise
train.AdamOptimizer(0.1).minimize(problem)

"""
PLOT
"""
U  = problem.get_U()
N  = len(U)

t = 1 + np.arange(N)
fig, axs = plt.subplots(3, 2, figsize=(10,13))

def plot_mean (ax, m, c='r', l='-'):
    ax.plot(t, m, c=c, linestyle=l)
    
def plot_dist (ax, m, s2, c='r'):
    plot_mean(ax, m, c=c, l='-')
    s = np.sqrt(s2)
    ax.fill_between(t, m+2*s, m-2*s, color=c, alpha=0.2)

# True states
model = models[0]
x0 = model.x0
Q  = model.P0
Xt, Yt = model.sample(x0, U)
for i in range(3):
    plot_mean(axs[i][0], Xt[:-1], c='k', l='--')
    plot_mean(axs[i][1], Yt[:,0], c='k', l='--')
    
for model,c in zip(models,['r','b','m']):
    # Predictions
    Xp, Pp = np.zeros(( N+1, 2 )), np.zeros(( N+1, 2, 2 ))
    Yp, Ep = np.zeros(( N, 1 )), np.zeros(( N, 1, 1 ))
    Xp[0], Pp[0] = model.x0, model.P0
    for n in range(N):
        Xp[n+1], Pp[n+1], Yp[n], Ep[n] = model.predict_dist(Xp[n], Pp[n], U[n])
    Xp, Pp = Xp[:-1], Pp[:-1]
    plot_dist(axs[0][0], Xp[:,0], Pp[:,0,0], c=c)
    plot_dist(axs[0][0], Xp[:,1], Pp[:,1,1], c=c)
    plot_dist(axs[0][1], Yp[:,0], Ep[:,0,0], c=c)

    # Filtered
    Xf, Pf = np.zeros(( N, 2 )), np.zeros(( N, 2, 2 ))
    Yf, Ef = np.zeros(( N, 1 )), np.zeros(( N, 1, 1 ))
    for n in range(N):
        Xf[n], Pf[n] = model.filter_x_dist(Yt[n], Xp[n], Pp[n])
        Yf[n], Ef[n] = model.predict_y_dist(Xf[n], Pf[n])
    plot_dist(axs[1][0], Xf[:,0], Pf[:,0,0], c=c)
    plot_dist(axs[1][0], Xf[:,1], Pf[:,1,1], c=c)
    plot_dist(axs[1][1], Yf[:,0], Ef[:,0,0], c=c)

    # Smoothed
    Xs, Ps = np.zeros(( N, 2 )), np.zeros(( N, 2, 2 ))
    Ys, Es = np.zeros(( N, 1 )), np.zeros(( N, 1, 1 ))
    Xs[-1], Ps[-1] = Xf[-1], Pf[-1]
    for n in range(N-1)[::-1]:
        Xs[n], Ps[n] = model.smooth_x_dist(Xf[n], Pf[n], Xs[n+1], Ps[n+1], U[n])
    for n in range(N):
        Ys[n], Es[n] = model.predict_y_dist(Xs[n], Ps[n])
    plot_dist(axs[2][0], Xs[:,0], Ps[:,0,0], c=c)
    plot_dist(axs[2][0], Xs[:,1], Ps[:,1,1], c=c)
    plot_dist(axs[2][1], Ys[:,0], Es[:,0,0], c=c)

plt.show()