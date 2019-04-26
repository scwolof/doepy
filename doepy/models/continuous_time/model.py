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
from scipy.integrate import odeint

from ..derivatives import LatentStateDerivatives
from ..statespacemodel import StateSpaceModel
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
		ode  = lambda t, x, args: self.f(x,*args)
		T    = self._get_time_steps(start=start_time)
		xk1  = odeint(ode, x, T, args=(args,), tfirst=True)[-1]
		yk   = np.matmul(self.H, x)
		return xk1, yk

	def _sample (self, x, u):
		us = mvn(u, self.u_covar)
		xy = self.predict(x, us)
		wk = mvn( np.zeros(self.num_states), self.step_length*self.x_covar )
		vk = mvn( np.zeros(self.num_meas), self.y_covar )
		return xy[0] + wk, xy[1] + vk
	
	def _ode_vector_merge (self, x, S, do=None, grad=False):
		assert_is_shape(x, (self.num_states,))
		assert_is_shape(S, (self.num_states, self.num_states))
		
		#if V is None:
		#	V = np.zeros(( self.num_states, self.num_states ))
		#assert_is_shape(V, (self.num_states, self.num_states))
		
		# Mean, covariance and input-output covariance
		xs = ( x[:,None], S ) #, V )
		
		if do is None:
			do = LatentStateDerivatives(self)
			do.dMdx = np.eye(self.num_states)
			do.dSds = np.eye(self.num_states)[:,None,:,None] \
			          * np.eye(self.num_states)[None,:,None,:]
		
		# dMdx, dMdu, dMdp
		xs += ( do.dMdx, do.dMdu )
		if self.num_param > 0:
			xs  += ( do.dMdp, )
		
		# dMds, dSdx, dSdu, dSdp
		xs += ( do.dMds.reshape(( self.num_states, -1 )),
				do.dSdx.reshape(( self.num_states, -1 )),
				do.dSdu.reshape(( self.num_states, -1 )) )
		if self.num_param > 0:
			xs  += ( do.dSdp.reshape(( self.num_states, -1 )), )
		
		# dSds, dVdx, dVdu, dVdp, dVds
		xs += ( do.dSds.reshape(( self.num_states, -1 )), )
		#		do.dVdx.reshape(( self.num_states, -1 )),
		#		do.dVdu.reshape(( self.num_states, -1 )) )
		#if self.num_param > 0:
		#	xs  += ( do.dVdp.reshape(( self.num_states, -1 )), )
		#xs += ( do.dVds.reshape(( self.num_states, -1 )), )
		
		#if not grad:
		#	X = np.concatenate(xs, axis=1)
		#	return X.T.flatten()
		
		X = np.concatenate(xs, axis=1)
		return X.T.flatten()
	
	def _ode_vector_unmerge (self, X, grad=False):
		# Mean x
		d = [1]
		# Covariance S
		d += [ self.num_states ]
		# Input-output covariance V
		#d += [ self.num_states ]
		
		# dMdx, dMdu, dMdp
		d += [self.num_states, self.num_inputs, self.num_param]
		# dMds
		d += [ self.num_states*self.num_states ]
		# dSdx, dSdu, dSdp
		d += [ self.num_states*self.num_states ]
		d += [ self.num_states*self.num_inputs ]
		d += [ self.num_states*self.num_param ]
		# dSds
		d += [ self.num_states*self.num_states*self.num_states ]
		# dVdx, dVdu, dVdp
		#d += [ self.num_states*self.num_states ]
		#d += [ self.num_states*self.num_inputs ]
		#d += [ self.num_states*self.num_param ]
		# dVds
		#d += [ self.num_states*self.num_states*self.num_states ]
			
		D = np.cumsum( d )
		X = X.reshape((D[-1], self.num_states)).T
		
		x = X[:,0];
		i = 0
		S = X[:,D[i]:D[i+1]]; i+= 1
		#V = X[:,D[i]:D[i+1]]; i+= 1
		
		do = LatentStateDerivatives(self)
		Dx = (self.num_states,)
		Du = (self.num_inputs,)
		Dp = (self.num_param,)
		
		do.dMdx = X[:,D[i]:D[i+1]]; i+= 1
		do.dMdu = X[:,D[i]:D[i+1]]; i+= 1
		if self.num_param > 0:
			do.dMdp = X[:,D[i]:D[i+1]]
		i+= 1
		do.dMds = X[:,D[i]:D[i+1]].reshape(Dx*3); i+= 1
		
		do.dSdx = X[:,D[i]:D[i+1]].reshape(Dx*3); i+= 1
		do.dSdu = X[:,D[i]:D[i+1]].reshape(Dx*2+(self.num_inputs,)); i+= 1
		if self.num_param > 0:
			do.dSdp = X[:,D[i]:D[i+1]].reshape(Dx*2+Dp)
		i+= 1
		do.dSds = X[:,D[i]:D[i+1]].reshape(Dx*4); i+= 1
		
		#do.dVdx = X[:,D[i]:D[i+1]].reshape(Dx*3); i+= 1
		#do.dVdu = X[:,D[i]:D[i+1]].reshape(Dx*2+Du); i+= 1
		#if self.num_param > 0:
		#	do.dVdp = X[:,D[i]:D[i+1]].reshape(Dx*2+Dp)
		#i+= 1
		#do.dVds = X[:,D[i]:D[i+1]].reshape(Dx*4); i+= 1
		
		return x, S, do # V, do

	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		do = LatentStateDerivatives(self)
		X = self._ode_vector_merge(xk, Sk, grad=grad)
		T = self._get_time_steps()
		X = odeint(self._x_dist_ode, X, T, args=(u,grad), tfirst=True)[-1]

		M, S, do = self._ode_vector_unmerge(X, grad=grad)
		S += np.matmul( do.dMdu, np.matmul(self.u_covar, do.dMdu.T) )
		if self.num_param > 0:
			S += np.matmul( do.dMdp, np.matmul(self.p_covar, do.dMdp.T) )
		#ret = (M, S, V) if cross_cov else (M, S)
		ret = (M, S, None) if cross_cov else (M, S)
		return ret if not grad else ret+(do,)
		
	def _x_dist_ode (self, t, X, U, grad=False):
		x, S, do = self._ode_vector_unmerge(X, grad=grad)
		
		# Input
		if not (isinstance(U, np.ndarray) or callable(U)):
			raise ValueError('Control input not array or callable')
		u = U if isinstance(U, np.ndarray) else U(t)
		
		x, S, domm = self._ode_moment_match(x, S, u, grad=grad)
		
		# Matrix multiplication
		def mul (s1, s2):
			a1, a2 = getattr(domm,s1), getattr(do,s2)
			t1 = {'M':'a'}.get(s1[1],'ab')
			t2 = {'x':'i', 's':'ij'}[s1[3]]
			t3 = {'x':'c', 'u':'c', 'p':'c'}.get(s2[3],'cd')
			return np.einsum( '%s%s,%s%s->%s%s'%(t1,t2,t2,t3,t1,t3), a1, a2 )
		
		dMdx = mul( 'dMdx', 'dMdx' ) + mul( 'dMds', 'dSdx' )
		dMds = mul( 'dMdx', 'dMds' ) + mul( 'dMds', 'dSds' )
		dMdu = domm.dMdu + mul( 'dMdx', 'dMdu' ) + mul('dMds','dSdu')
		if self.num_param > 0:
			dMdp = domm.dMdp + mul('dMdx','dMdp') + mul('dMds','dSdp')
		
		dSdx = mul( 'dSdx', 'dMdx' ) + mul( 'dSds', 'dSdx' )
		dSds = mul( 'dSdx', 'dMds' ) + mul( 'dSds', 'dSds' )
		dSdu = domm.dSdu + mul( 'dSdx', 'dMdu' ) + mul( 'dSds', 'dSdu' )
		if self.num_param > 0:
			dSdp = domm.dSdp + mul( 'dSdx', 'dMdp' ) + mul( 'dSds', 'dSdp' )
		
		#dVdx = mul( 'dVdx', 'dMdx' ) + mul( 'dVds', 'dSdx' )
		#dVds = mul( 'dVdx', 'dMds' ) + mul( 'dVds', 'dSds' )
		#dVdu = domm.dVdu + mul( 'dVdx', 'dMdu' ) + mul( 'dVds', 'dSdu' )
		#if self.num_param > 0:
		#	dVdp = domm.dVdp + mul( 'dVdx', 'dMdp' ) + mul( 'dVds', 'dSdp' )
		
		do.dMdx = dMdx
		do.dMds = dMds
		do.dMdu = dMdu
		do.dSdx = dSdx
		do.dSds = dSds
		do.dSdu = dSdu
		#do.dVdx = dVdx
		#do.dVds = dVds
		#do.dVdu = dVdu
		if self.num_param > 0:
			do.dMdp = dMdp
			do.dSdp = dSdp
			#do.dVdp = dVdp
		
		# Total
		return self._ode_vector_merge(x, S, do, grad=grad)
	
	def _get_time_steps (self, start=0., stop=None):
		if stop is None:
			stop = start + self.step_length
		assert start < stop, 'Start time (%f) >= stop time (%f)'%(start,stop)
		return np.linspace(start, stop, self.time_steps)