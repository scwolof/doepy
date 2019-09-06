"""
 Test functions for later implementation
 """

import numpy as np
from scipy.integrate import solve_ivp

def model(x,u,p):
	return p*x - 0.7*u

def model_disc(x0,p,u,t0,tf,nt):
	dt = (tf-t0)/nt

	x = np.zeros(nt);
	dx = np.zeros(nt);

	x[0] = x0
	dx[0] = model(x0,u,p);

	for i in range(1,nt):
		x[i] = x[i-1] + dt*model(x[i-1],u,p)
		dx[i] = model(x[i],u,p);

	return x, dx

def model_disc_RK2(x0,p,u,t0,tf,nt):
	dt = (tf-t0)/nt

	x = np.zeros(nt);
	dx = np.zeros(nt);

	x[0] = x0
	dx[0] = model(x0,u,p);

	for i in range(1,nt):
		x[i] = x[i-1] + dt*(0.25*dx[i-1] + 0.75*model(x[i-1]+(2/3)*dx[i-1]*dt,u,p))
		dx[i] = model(x[i],u,p);

	return x, dx

def model_ode(x0,p,u,t0,tf,npts):
	Tp = np.linspace(t0,tf,npts)
	T = (t0,tf)
	args = (u, p)
	ode  = lambda t, x: model(x,*args)
	sol  = solve_ivp(ode, T,  np.asarray([x0]), t_eval=Tp, rtol=1e-8, atol=1e-8)
	t = sol['t']
	x = sol['y']
	return t,x

def interp(x,dx,tp):
	npts = len(tp)
	dt = tp[1]-tp[0]
	mat_A = np.zeros((2*npts,2*npts))
	mat_B = np.zeros(2*npts)

	for i in range(0,npts):
		exp_i  = lambda t: np.exp(-(((t-tp[i])/dt)**2))

		mat_B[2*i] = x[i]
		mat_B[2*i+1] = dx[i]
		for j in range(0,npts):
			k_i = exp_i(tp[j])
			mat_A[2*j,2*i] = tp[j]*k_i;
			mat_A[2*j,2*i+1] = k_i;
			mat_A[2*j+1,2*i] = k_i-(2/dt)*k_i*tp[j]*(tp[j]-tp[i]);
			mat_A[2*j+1,2*i+1] = -(2/dt)*k_i*(tp[j]-tp[i]);
	return mat_A, mat_B

def sim_fun(coef, t, tp):
	f_res = 0
	dt = tp[1]-tp[0]
	for i in range(0,len(tp)):
		f_res += (coef[2*i]*t + coef[2*i+1])*np.exp(-(((t-tp[i])/dt)**2))

	return f_res

def sim_u(coef, t, tp):
	f_res = np.zeros(len(t)-1)
	t_res = np.zeros(len(t)-1)
	k = 0

	for i in range(0,len(tp)-1):
		
		ti = t[np.where(np.logical_and(t>=tp[i], t<tp[i+1]))]
		
		for j in ti:
		
			f_res[k] = coef[i,0] +  coef[i,1]*(j-tp[i]) + coef[i,2]*((j-tp[i])**2) + coef[i,3]*((j-tp[i])**3)
			t_res[k] = j
			k +=1

	return t_res, f_res
