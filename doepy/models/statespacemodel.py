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
from scipy.linalg import block_diag

from .model import Model
from .derivatives import LatentStateDerivatives, MeasurementDerivatives
from ..utils import assert_symmetric_matrix, assert_is_shape, assert_not_none

class StateSpaceModel (Model):
	def __init__ (self, candidate_model):
		if candidate_model.num_meas is None:
			candidate_model.num_meas = candidate_model.H.shape[0]
		super().__init__(candidate_model)

		self.H = candidate_model.H
		self.num_states = self.H.shape[1]

		# Process noise
		self.x_covar = candidate_model.x_covar
		if self.x_covar is None:
			self.x_covar = np.zeros(( self.num_states, self.num_states ))

		# Initial states
		self.x0       = candidate_model.x0
		self.x0_covar = candidate_model.x0_covar
		if self.x0_covar is None:
			self.x0_covar = np.zeros(( self.num_states, self.num_states ))

	"""
	Measurement matrix
	"""
	@property
	def H (self):
		return self._H 
	@H.setter
	def H (self, H):
		assert isinstance(H, np.ndarray)
		assert H.ndim == 2
		self._H = H.copy()

	"""
	Process noise covariance matrix
	"""
	@property
	def x_covar (self):
		return self._x_covar
	@x_covar.setter
	def x_covar (self, x_covar):
		assert_symmetric_matrix(x_covar)
		self._x_covar = x_covar.copy()

	"""
	Initial state mean
	"""
	@property
	def x0 (self):
		return self._x0 
	@x0.setter
	def x0 (self, x0):
		assert_is_shape(x0, (self.num_states,))
		self._x0 = x0.copy()

	"""
	Initial state covariance
	"""
	@property
	def x0_covar (self):
		return self._x0_covar 
	@x0_covar.setter
	def x0_covar (self, x0_covar):
		if x0_covar is None:
			x0_covar = np.zeros(( self.num_states, self.num_states ))
		assert_symmetric_matrix(x0_covar)
		self._x0_covar = x0_covar.copy()

	"""
	Latent state constraints
	- For surrogate models that want to enforce state constraints in order
	  not to deviate too far from training data.
	"""
	def initialise_x_constraints (self):
		pass

	def update_x_constraints (self, x, s, dxdU, dsdU, step=None):
		pass

	def get_x_constraints (self):
		return None

	def num_x_constraints (self):
		return 0

	"""
	Mean and covariance of input distribution
	"""
	def get_input_mean_and_cov (self, x, S, u, concatenate=False):
		assert_is_shape(x, (self.num_states,))
		assert_is_shape(S, (self.num_states,self.num_states))
		assert_is_shape(u, (self.num_inputs,))

		mean = ( x, u )
		covs = ( S, self.u_covar )

		if self.num_param > 0:
			assert_not_none(self.p_mean, '%s:p_mean'%self.name)
			assert_not_none(self.p_covar, '%s:p_covar'%self.name)
			mean += ( self.p_mean, )
			covs += ( self.p_covar, )

		if concatenate:
			mean = np.concatenate( mean )
		cov = block_diag(*covs)

		return mean, cov

	"""
	Retrieve latent state deriv object from approx. infererence deriv object
	"""
	def get_latent_state_derivatives (self, domm, cross_cov=True, hessian=False):
		do = LatentStateDerivatives(self)
		D  = self.num_states
		dn = D + self.num_inputs
		do.dMdx = domm.dMdx[:,:D]
		do.dMdu = domm.dMdx[:,D:dn]
		do.dMds = domm.dMds[:,:D,:D]
		do.dSdx = domm.dSdx[:,:,:D]
		do.dSdu = domm.dSdx[:,:,D:dn]
		do.dSds = domm.dSds[:,:,:D,:D]
		if cross_cov:
			do.dVdx = domm.dVdx[:D,:,:D]
			do.dVdu = domm.dVdx[:D,:,D:dn]
			do.dVds = domm.dVds[:D,:,:D,:D]
		if self.num_param > 0:
			do.dMdp = domm.dMdx[:,dn:]
			do.dSdp = domm.dSdx[:,:,dn:]
			if cross_cov:
				do.dVdp = domm.dVdx[:D,:,dn:]
		if not hessian:
			return do
		do.dMdxx = domm.dMdxx[:,:D,:D]
		do.dMdxu = domm.dMdxx[:,:D,D:dn]
		do.dMduu = domm.dMdxx[:,D:dn,D:dn]
		if self.num_param > 0:
			do.dMdxp = domm.dMdxx[:,:D,dn:]
			do.dMdup = domm.dMdxx[:,D:dn,dn:]
			do.dMdpp = domm.dMdxx[:,dn:,dn:]
		return do

	"""
	Function calls
	"""
	def predict (self, x0, U, **kwargs):
		"""
		Deterministic prediction from model
		    x_{k+1} = f( x_k, u_k )
		        y_k = H * x_k

		If U.ndim == 1, one-step prediction
		If U.ndim == 2, multi-step prediction
		"""
		if x0 is None:
			x0 = self.x0
		if self.num_param > 0:
			assert_not_none(self.p_mean,'%s:p_mean'%self.name)

		if U.ndim == 1:
			return self._predict(x0, U, **kwargs)

		n = len(U)
		X = np.zeros(( n+1, self.num_states ))
		Y = np.zeros((   n, self.num_meas ))

		X[0] = x0
		for k in range( n ):
			X[k+1], Y[k] = self._predict(X[k], U[k], **kwargs)
		return X, Y

	def _predict (self, x0, u, **kwargs):
		raise NotImplementedError

	def sample (self, x0, U, initial_uncertainty=False):
		"""
		Stochastic model simulation
			x_{k+1} = f( x_k, u_k ) + w_k
			    y_k = H * x_k + v_k

		If U.ndim == 1, one-step prediction
		If U.ndim == 2, multi-step prediction
		"""
		if x0 is None:
			x0 = self.x0
		x0 = x0 if not initial_uncertainty else mvn(x0, self.x0_covar)

		if U.ndim == 1:
			return self._sample(x0, U)

		n = len(U)
		X = np.zeros(( n+1, self.num_states ))
		Y = np.zeros((   n, self.num_meas ))

		X[0] = x0
		for k in range(n):
			X[k+1], Y[k] = self._sample(X[k], U[k])
		return X, Y

	def predict_x_dist (self, xk, Sk, U, cross_cov=False, grad=False):
		"""
		Input state posterior mean xk and variance Sk, and controls U 
			p( x_k | y_{1 : k} ) = N( xk, Sk )
		Outputs mean and variance of state prediction
			p( x_{k+1} | y_{1 : k} ) = int[ f(x, u) * N(x | xk, Sk) ] dx

		If U.ndim == 1, one-step prediction.
		If U.ndim == 2, multi-step prediction
		"""
		if U.ndim == 1:
			return self._predict_x_dist(xk, Sk, U, cross_cov=cross_cov, grad=grad)

		n = len(U)
		X = np.zeros(( n+1, self.num_states ))
		S = np.zeros(( n+1, self.num_states, self.num_states ))

		X[0] = xk
		S[0] = Sk
		if not grad:
			for k in range(n):
				X[k+1], S[k+1] = self._predict_x_dist(X[k], S[k], U[k])
			return X, S

		do = LatentStateDerivatives(self, num_test_points=n)
		for k in range(n):
			X[k], S[k], dok = self._predict_x_dist(X[k], S[k], U[k], grad=True)
			do.insert(dok,k)
		return X, S, do

	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		# Implemented in children classes
		raise NotImplementedError

	def predict_z_dist (self, x, s, grad=False):
		"""
		Input: latent state mean x and covariance s
			p( x_k | y_{1 : T} ) = N( x, s )
		Output: observed state mean and variance of observation
			p( z_k | y_{1 : T} ) = N( Z, S )

		If x.ndim == 1, one-step prediction
		If x.ndim == 2, multi-step prediction
		"""
		if x.ndim == 1:
			return self._predict_z_dist(x, s, grad=grad)

		n = len(x)
		Z = np.zeros(( n, self.num_meas ))
		S = np.zeros(( n, self.num_meas, self.num_meas ))
		if not grad:
			for k in range(n):
				Z[k], S[k] = self._predict_z_dist(x[k], s[k], grad=grad)
			return Z, S

		do = MeasurementDerivatives(self, n=n)
		for k in range(n):
			Z[k], S[k], dok = self._predict_y_dist(x[k], s[k], grad=True)
			do.insert(dok,k)
		return Z, S, do

	def _predict_z_dist (self, x, s, grad=False):
		Z   = np.matmul(self.H, x)
		S   = np.matmul(self.H, np.matmul(s, self.H.T) )
		ret = (Z, S)
		if grad:
			do = MeasurementDerivatives(self)
			do.dMdx = self.H
			for e1 in range(  self.num_meas ):
				for e2 in range(  self.num_meas ):
					do.dSds[e1,e2] = self.H[e1][:,None] * self.H[e2][None,:]
			ret += (do,)
		return ret

	def predict_y_dist (self, x, s, grad=False):
		"""
		Input: latent state mean x and covariance s
			p( x_k | y_{1 : T} ) = N( x, s )
		Output: observed state mean and variance of observation
			p( y_k | y_{1 : T} ) = N( Y, S )

		If x.ndim == 1, one-step prediction
		If x.ndim == 2, multi-step prediction
		"""
		res = self.predict_z_dist(x, s, grad=grad)
		if grad:
			Y, S, do = res
		else:
			Y, S = res

		R  = self.y_covar if x.ndim == 1 else self.y_covar[None,:,:]
		S += R

		if grad:
			return Y, S, do
		return Y, S

	def filter (self, yk, x, s):
		"""
		Input observation yk, prediction mean m and variance s 
			p( x_k | y_{1 : k-1} ) = N( x, s )
		Outputs state posterior mean xk and variance sk 
			p( x_k | y_{1 : k} ) = N( xk, sk )

		yk : [ (n), E ]
		x  : [ (n), D ]
		s  : [ (n), D, D ]
		"""
		assert yk.ndim == x.ndim

		if yk.ndim == 1:
			return self._filter(yk, x, s)

		n = len(yk)
		X = np.zeros(( n, self.num_states ))
		S = np.zeros(( n, self.num_states, self.num_states ))
		for k in range(n):
			X[k], S[k] = self._filter(yk[k], x[k], s[k])
		return X, S

	def _filter (self, yk, x, s):
		sH = np.matmul(s, self.H.T)
		K  = np.matmul(sH, np.linalg.inv(np.matmul(self.H, sH) + self.y_covar))
		xk = x + np.matmul(K, yk - np.matmul(self.H, x))
		sk = s - np.matmul(K, sH.T)
		return xk, sk

	def predict_filter (self, Y, x0, x0_covar, U):
		"""
		Filter sequence, based on observations Y, controls U,
		and with prediction p(x_1) ~ N(x0, x0_covar)

		Y        : [ n, E ]       ( y_1, ..., y_n )
		x0       : [ D, ]
		x0_covar : [ D, D ]
		U        : [ n, D_U ]     ( u_1, ..., u_{n-1} )

		Outputs
		x  : [ n+1, D ]       ( x_1, ..., x_n )
		s  : [ n+1, D, D]
		"""
		n = len(Y)
		x = np.zeros(( n, self.num_states ))
		s = np.zeros(( n, self.num_states, self.num_states ))
		M, S = x0, x0_covar
		for k in range( n ):
			x[k], s[k] = self._filter(Y[k], M, S)
			if k < n-1:
				M, S = self.predict_x_dist(x[k], s[k], U[k])
		return x, s

	def smooth (self, X, S, U):
		"""
		Iteratively smooths sequences X = [x_1, ..., x_n] and 
		S = [S_1, ..., S_n] with controls U = [u_1, ..., u_n]
		"""
		n  = len(X)
		Xs = np.zeros(( n, self.num_states ))
		Ss = np.zeros(( n, self.num_states, self.num_states ))
		Xs[-1] = X[-1]
		Ss[-1] = S[-1]
		for k in np.arange(1, n)[::-1]:
		    Xs[k-1],Ss[k-1] = self._smooth(X[k-1], S[k-1], Xs[k], Ss[k], U[k-1])
		return Xs, Ss

	def _smooth (self, xk, Sk, xk1, Sk1, uk):
		"""
		Inputs xk, Sk, xk1, S_{k+1}, uk
			p( x_k | y_{1:k} ) = N( xk, Sk )
			p( x_{k+1} | y_{1:T}, uk ) = N( xk1, Sk1 )

		Produces smoothed x_k, S_k
			p( x_k | y_{1:T} ) = N( xs, Ss )
		"""
		m, S, V = self.predict_x_dist(xk, Sk, uk, cross_cov=True)
		J  = np.matmul( V, np.linalg.inv(S) )
		xs = xk + np.matmul( J, xk1 - m )
		Ss = Sk + np.matmul( J, np.matmul( Sk1 - S, J.T ) )
		return xs, Ss