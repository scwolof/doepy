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
from numpy.random import multivariate_normal as mvn
from scipy.integrate import solve_ivp

from ..derivatives import LatentStateDerivatives
from ..statespacemodel import StateSpaceModel
from ...derivatives import Derivatives
from ...utils import assert_is_shape

class ctModel (StateSpaceModel):
	def __init__ (self, candidate_model):
		"""
		Model:
			d x / dt = f( x(t), u, p )  +  w(t),   w(t) ~ N(0, x_covar)
				 z_k = H * x_k
				 y_k = z_k  +  v_k,                v_k ~ N(0, y_covar)
		with 
			x_0 ~ N(x0, x0_covar), 
			u_k ~ N(u_k, u_covar), 
			p   ~ N(p, p_covar)
			u_k of dimension num_inputs
		"""
		super().__init__(candidate_model)
		self.step_length = candidate_model.step_length
		self.time_steps  = candidate_model.time_steps
		if self.time_steps is None:
			self.time_steps = 101

	def _predict (self, x, u, start_time=0.):
		args = (u,) if self.num_param < 1 else (u, self.p_mean)
		ode  = lambda t, x: self.f(x,*args)
		T    = self._get_time_steps(start=start_time)
		T    = (T[0], T[-1])
		xk1  = solve_ivp(ode, T, x)['y'][:,-1]
		yk   = np.matmul(self.H, x)
		return xk1, yk

	def _sample (self, x, u):
		us = mvn(u, self.u_covar)
		xy = self.predict(x, us)
		wk = mvn( np.zeros(self.num_states), self.step_length*self.x_covar )
		vk = mvn( np.zeros(self.num_meas), self.y_covar )
		return xy[0] + wk, xy[1] + vk
	
	def _ode_mean_var_merge (self, x, S, u):
		mean = (x, u) 
		var  = (S, self.u_covar)
		if self.num_param > 0: 
			mean += ( self.p_mean, )
			var  += ( self.p_covar, )
		mean = np.concatenate(mean)
		var  = block_diag(*var)
		return mean, var
	
	def _ode_mean_var_unmerge (self, mean, var, do=None, grad=False):
		Dx, Du, Dp = self.num_states, self.num_inputs, self.num_param
		x, S = mean[:Dx], var[:Dx,:Dx]
		domm = LatentStateDerivatives(self)
		if grad:
			domm.dMdx = do.dMdx[:Dx,:Dx]
			domm.dMdu = do.dMdx[:Dx,Dx:(Dx+Du)]
			domm.dMds = do.dMds[:Dx,:Dx,:Dx]
			domm.dSdx = do.dSdx[:Dx,:Dx,:Dx]
			domm.dSdu = do.dSdx[:Dx,:Dx,Dx:(Dx+Du)]
			domm.dSds = do.dSds[:Dx,:Dx,:Dx,:Dx]
			if Dp > 0:
				domm.dMdp = do.dMdx[:Dx,-Dp:]
				domm.dSdp = do.dSdx[:Dx,:Dx,-Dp:]
		return x, S, domm

	def _ode_vector_merge (self, mean, var, do=None, grad=False):
		Dx, Du, Dp = self.num_states, self.num_inputs, self.num_param
		D = Dx + Du + Dp
		assert_is_shape(mean, (D,))
		assert_is_shape(var, (D, D))

		# Mean, covariance and input-output covariance
		xs = ( mean[:,None], var )
		
		if not grad:
			return np.concatenate(xs, axis=1).T.flatten()

		if do is None:
			do = Derivatives(D, num_states=D)
			do.dMdx = np.eye(D)
			do.dSds = np.eye(D)[:,None,:,None] * np.eye(D)[None,:,None,:]
			
		# dMdx
		xs += ( do.dMdx, )
		# dMds, dSdx
		xs += ( do.dMds.reshape(( D, -1 )),
				do.dSdx.reshape(( D, -1 )) )
		# dSds
		xs += ( do.dSds.reshape(( D, -1 )), )

		return np.concatenate(xs, axis=1).T.flatten()

	def _ode_vector_unmerge (self, X, grad=False):
		Dx, Du, Dp = self.num_states, self.num_inputs, self.num_param
		D = Dx + Du + Dp
		# Mean x
		d = [1]
		# Covariance S
		d += [ D ]

		if grad:
			# dMdx, dMds, dSdx, dSds
			d += [ D, D*D, D*D, D*D*D ]

		Ds = np.cumsum( d )
		X  = X.reshape((Ds[-1], D)).T

		x  = X[:,0]
		i  = 0
		S  = X[:,Ds[i]:Ds[i+1]]; i+= 1
		do = Derivatives(D, num_states=D)
		if not grad:
			return x, S, do

		do.dMdx = X[:,Ds[i]:Ds[i+1]]; i+= 1
		do.dMds = X[:,Ds[i]:Ds[i+1]].reshape((D,D,D)); i+= 1
		do.dSdx = X[:,Ds[i]:Ds[i+1]].reshape((D,D,D)); i+= 1
		do.dSds = X[:,Ds[i]:Ds[i+1]].reshape((D,D,D,D)); i+= 1

		return x, S, do
	
	def _get_ode_mean_grad (self, do, grad=False):
		Dx, Du, Dp = self.num_states, self.num_inputs, self.num_param
		D    = np.cumsum([Du+Dp, Dx])
		domm = Derivatives(D[1], num_states=D[1])
		
		dMs = (do.dMdx, do.dMdu)
		if self.num_param > 0:
			dMs += (do.dMdp,)
		dMs = np.concatenate(dMs, axis=1)
		domm.dMdx = np.vstack(( dMs, np.zeros((D[0], D[1])) ))
		
		if not grad:
			return domm
		
		ddM = np.zeros((D[1], D[1], D[1]))
		ddM[:Dx,:Dx,:Dx] = do.dMdxx
		ddM[:Dx,:Dx,Dx:Dx+Du] = do.dMdxu
		ddM[:Dx,Dx:Dx+Du,:Dx] = np.transpose(do.dMdxu,[0,2,1])
		ddM[:Dx,Dx:Dx+Du,Dx:Dx+Du] = do.dMduu
		if Dp > 0:
			ddM[:Dx,:Dx,-Dp:] = do.dMdxp
			ddM[:Dx,-Dp:,:Dx] = np.transpose(do.dMdxp, [0,2,1])
			ddM[:Dx,-Dp:,-Dp:] = do.dMdpp
			ddM[:Dx,Dx:Dx+Du,-Dp:] = do.dMdup
			ddM[:Dx,-Dp:,Dx:Dx+Du] = np.transpose(do.dMdup, [0,2,1])
		domm.dMdxx = ddM
		
		return domm

	def _ode_moment_match (self, M, S, grad=False):
		Dx, Du, Dp = self.num_states, self.num_inputs, self.num_param
		D = self.num_states + self.num_inputs + self.num_param
		
		input_mean = (M[:Dx], M[Dx:(Dx+Du)])
		if Dp > 0:
			input_mean += ( M[(Dx+Du):], )
			
		dx, do = self.f( *input_mean, grad=True )
		do = self._get_ode_mean_grad(do, grad=grad) 
		dM = np.zeros(D)
		dM[:self.num_states] = dx
		
		dS  = np.zeros((D,D))
		dS[:Dx, :Dx] = self.x_covar
		dS += np.matmul(do.dMdx, S) + np.matmul(S, do.dMdx.T)
		if not grad:
			return dM, dS, do

		I = np.eye(D)[:,None,:,None] * np.eye(D)[None,:,None,:]
		do.dSdx  = np.einsum("kj,ijn->kin", S, do.dMdxx)
		do.dSdx += np.transpose(do.dSdx, [1,0,2])
		do.dSds  = np.einsum('ij,jkmn->ikmn',do.dMdx,I)
		do.dSds += np.transpose(do.dSds, [1,0,3,2])

		return dM, dS, do
		
	def _x_dist_ode (self, t, X, grad=False):
		mean, var, do = self._ode_vector_unmerge(X, grad=grad)

		x, S, domm = self._ode_moment_match(mean, var, grad=grad)
		
		if not grad:
			self._ode_vector_merge(x, S, do, grad=False)
		
		# Matrix multiplication
		def mul (s1, s2):
			a1, a2 = getattr(domm,s1), getattr(do,s2)
			t1 = {'M':'a'}.get(s1[1],'ab')
			t2 = {'x':'i', 's':'ij'}[s1[3]]
			t3 = {'x':'c', 'u':'c', 'p':'c'}.get(s2[3],'cd')
			return np.einsum( '%s%s,%s%s->%s%s'%(t1,t2,t2,t3,t1,t3), a1, a2 )
	
		dMdx = mul( 'dMdx', 'dMdx' ) + mul( 'dMds', 'dSdx' )
		dMds = mul( 'dMdx', 'dMds' ) + mul( 'dMds', 'dSds' )
		dSdx = mul( 'dSdx', 'dMdx' ) + mul( 'dSds', 'dSdx' )
		dSds = mul( 'dSdx', 'dMds' ) + mul( 'dSds', 'dSds' )
		
		do.dMdx = dMdx
		do.dMds = dMds
		do.dSdx = dSdx
		do.dSds = dSds
		return self._ode_vector_merge(x, S, do, grad=True)

	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		if cross_cov:
			raise NotImplementedError(
				'Cross covariance not implemented for continuous time models')
		mean, var = self.get_input_mean_and_cov(xk, Sk, u, concatenate=True)

		ode = lambda t, x: self._x_dist_ode(t, x, grad=grad)
		X   = self._ode_vector_merge(mean, var, grad=grad)
		T   = self._get_time_steps()
		T   = (T[0], T[-1])
		X   = solve_ivp(ode, T, X)['y'][:,-1]

		M, S, do = self._ode_vector_unmerge(X, grad=grad)
		M, S, do = self._ode_mean_var_unmerge(M, S, do, grad=grad)
		return (M, S) if not grad else (M, S, do)

	def _get_time_steps (self, start=0., stop=None):
		if stop is None:
			stop = start + self.step_length
		assert start < stop, 'Start time (%f) >= stop time (%f)'%(start,stop)
		return np.linspace(start, stop, self.time_steps)