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

from ..model import Model
from ...utils import assert_symmetric_matrix, assert_is_shape

class dtModel (Model):
	def __init__ (self, candidate_model):
		"""
		Model:
			x_{k+1} = f( x_k, u_k, p )  +  w_k,   w_k ~ N(0, x_covar)
			    z_k = H * x_k
			    y_k = z_k  +  v_k,                v_k ~ N(0, y_covar)
		with 
			x_0 ~ N(x0, x0_covar), 
			u_k ~ N(u_k, u_covar), 
			p   ~ N(p, p_covar)
			u_k of dimension num_inputs
		"""
		if candidate_model.num_meas is None:
			candidate_model.num_meas = candidate_model.H.shape[0]
		super().__init__(candidate_model)

		self.H = candidate_model.H
		self.num_states = self.H.shape[0]

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
	Function calls
	"""
	def predict (self, x0, U):
		"""
		Deterministic prediction from model
			x_{k+1} = f( x_k, u_k )
			    y_k = H * x_k

		If U.ndim == 1, one-step prediction
		If U.ndim == 2, multi-step prediction
		"""
		if U.ndim == 1:
			return self._predict(x0, U)

		n = len(U)
		X = np.zeros(( n+1, self.num_states ))
		Y = np.zeros((   n, self.num_meas ))

		X[0] = x0
		for k in range( n ):
			X[k+1], Y[k] = self._predict(X[k], U[k])
		return X, Y

	def _predict (self, x, u):
		xk1 = self.f(x, u)
		yk  = np.matmul(self.H, x)
		return xk1, yk

	def sample (self, x0, U, initial_uncertainty=False):
		"""
		Stochastic model simulation
			x_{k+1} = f( x_k, u_k ) + w_k
			    y_k = H * x_k + v_k

		If U.ndim == 1, one-step prediction
		If U.ndim == 2, multi-step prediction
		"""
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

	def _sample (self, x, u):
		us = mvn(u, self.u_covar)
		xy = self.predict(x, us)
		wk = mvn( np.zeros(self.num_states), self.x_covar )
		vk = mvn( np.zeros(self.num_meas), self.y_covar )
		return xy[0] + wk, xy[1] + vk

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
		dXdx = np.zeros(( n, self.num_states, self.num_states ))
		dXds = np.zeros(( n, self.num_states, self.num_states, self.num_states ))
		dXdu = np.zeros(( n, self.num_states, self.num_inputs ))
		dSdx = np.zeros(( n, self.num_states, self.num_states, self.num_states ))
		dSds = np.zeros([ n ] + [ self.num_states ] * 4 )
		dSdu = np.zeros(( n, self.num_states, self.num_states, self.num_inputs ))
		for k in range(n):
			X[k], S[k], dXdx[k], dXds[k], dXdu[k], dSdx[k], dSds[k], dSdu[k] \
			                = self._predict_x_dist(X[k], S[k], U[k], grad=True)
		return X, S, dXdx, dXds, dXdu, dSdx, dSds, dSdu

	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		# Implemented in children classes (e.g. LinearModel)
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
		dYdx = np.zeros(( n, self.num_meas, self.num_states ))
		dYds = np.zeros(( n, self.num_meas, self.num_states, self.num_states ))
		dSdx = np.zeros(( n, self.num_meas, self.num_meas, self.num_states ))
		dSds = np.zeros( [n] + [self.num_meas]*2 + [self.num_states]*2 )
		for k in range(n):
			Z[k], S[k], dYdx[k], dYds[k], dSdx[k], dSds[k] \
			                = self._predict_y_dist(x[k], s[k], grad=True)
		return Z, S, dYdx, dYds, dSdx, dSds

	def _predict_z_dist (self, x, s, grad=False):
		Z = np.matmul(self.H, x)
		S = np.matmul(self.H, np.matmul(s, self.H.T) )
		if grad:
			dYdm = self.H
			dYds = np.zeros( [self.num_meas] + [self.num_states]*2 )
			dSdx = np.zeros( [self.num_meas]*2 + [self.num_states] )
			dSds = np.zeros( [self.num_meas]*2 + [self.num_states]*2 )
			for e1 in range(  self.num_meas ):
				for e2 in range(  self.num_meas ):
					dSds[e1,e2] = self.H[e1][:,None] * self.H[e2][None,:]
			return Z, S, dYdm, dYds, dSdx, dSds
		return Z, S

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
			Y, S, dYdx, dYds, dSdx, dSds = res
		else:
			Y, S = res

		if x.ndim == 1:
			S = S + self.y_covar
		else:
			S = S + self.y_covar[None,:,:]

		if grad:
			return Y, S, dYdx, dYds, dSdx, dSds
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

