"""
MIT License

Copyright (c) 2019 Simon Olofsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from GPy.kern import RBF, Exponential, Matern32, Matern52, RatQuad, Cosine 

def d_pred_d_x (gps, xnew, diag=False):
	E,D  = len(gps), len(xnew)
	dm   = np.zeros((E,D))
	ds   = np.zeros((E,E,D)) if diag else np.zeros((E,D))
	for e,gp in enumerate(gps):
		tmp   = gp.predictive_gradients(xnew[None,:])
		dm[e] = tmp[0][0,:,0]
		if diag:
			ds[e,e] = tmp[1][0]
		else:
			ds[e] = tmp[1][0]
	return dm, ds

def d2_m_d_x2 (gps, xnew):
	E, D = len(gps), len(xnew)
	ddm  = np.zeros((E,D,D))
	for e,gp in enumerate(gps):
		ddm[e] = gradients_XX(gp, xnew)
	return ddm

def gradients_XX (gp, xnew):
	"""
	Large parts of this function is copied from the GPy package 
	
	https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/stationary.py
	Copyright (c) 2012, GPy authors (see AUTHORS.txt).
	Licensed under the BSD 3-clause license (see LICENSE.txt)
	"""
	I  =  gp.kern.active_dims
	D  =  len( I )
	X  =  xnew[None,I]
	X2 =  gp._predictive_variable[:,I]
	B  = -gp.posterior.woodbury_vector.T
	r  =  gp.kern._scaled_dist(X, X2)
	N  =  gp._predictive_variable.shape[0]

	invdist  = gp.kern._inv_dist(X, X2)
	invdist2 = invdist**2
	dL_dr    = gp.kern.dK_dr(r)
	dL_drdr  = dK2_drdr(gp.kern, r)
	tmp1 = dL_dr * invdist
	tmp2 = dL_drdr * invdist2
	l2   = np.ones(D) * gp.kern.lengthscale**2

	tmp1[invdist2==0.] -= gp.kern.variance

	dist  = X[:,None,:] - X2[None,:,:]
	dist  = (dist[:,:,:,None]*dist[:,:,None,:])
	I     = np.ones((1,N,D,D)) * np.eye(D)
	_grad = (((B*(tmp1*invdist2-tmp2))[:,:,None,None]*dist)/l2[None,None,:,None]
	        - (B*tmp1)[:,:,None,None] * I)/l2[None,None,None,:]
	_grad = np.sum( _grad[0], axis=0)
	Dtot  = gp._predictive_variable.shape[1]
	grad  = np.zeros(( Dtot, Dtot ))
	for i,ad1 in enumerate(gp.kern.active_dims):
		for j,ad2 in enumerate(gp.kern.active_dims):
			grad[ad1,ad2] = _grad[i,j]
	return grad


def dK2_drdr (kern, r):
	d = { RBF:         rbf_dK2_drdr,
	      Exponential: exponential_dK2_drdr,
	      Matern32:    matern32_dK2_drdr,
	      Matern52:    matern52_dK2_drdr,
	      Cosine:      cosine_dK2_drdr,
	      RatQuad:     ratquad_dK2_drdr }
	t = type(kern)
	if t not in d:
		if 'dK2_drdr' in dir(kern):
			return kern.dK2_drdr(r)
		raise ValueError('Kernel does not implement dK2_drdr function')
	return d[type(kern)](kern, r)

def rbf_dK2_drdr (kern, r):
	return kern.dK2_drdr(r)

def exponential_dK2_drdr (kern, r):
	return kern.K_of_r(r)

def matern32_dK2_drdr (kern, r):
	ar = np.sqrt(3.)*r
	return 3. * kern.variance * (ar - 1) * np.exp(-ar)

def matern52_dK2_drdr (kern, r):
	ar = np.sqrt(5)*r 
	return 5./3. * kern.variance * (ar**2 - ar - 1) * np.exp(-ar)

def cosine_dK2_drdr (kern, r):
	return -kern.K_of_r(r)

def ratquad_dK2_drdr (kern, r):
	r2  = np.square(r)
	lp  = np.log1p(r2 / 2.)
	a   = (kern.power + 1) * lp
	dr1 = -kern.variance * kern.power * np.exp(-a)
	a  += lp
	dr2 = kern.variance * kern.power * (kern.power + 1) * r2 * np.exp(-a)
	return dr1 + dr2