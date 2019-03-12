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

from ..utils import is_symmetric_matrix, is_pos_def

class Model:
	def __init__ (self, candidate_model):
		"""
		Model:
			x_{k+1} = f( x_k, u_k )  +  w_k,   w_k ~ N(0, Q)
			    y_k = H * x_k  +  v_k,         v_k ~ N(0, R)
		with 
			x_0 ~ N(x0, S_x0), u_k ~ N(u_k, S_u)
			u_k of dimension num_inputs
		"""
		self.name = candidate_model.name
		if self.name is None:
			self.name = 'Model' + np.random.randint(1000)

		self.f = candidate_model.f
		self.H = candidate_model.H
		self.Q = candidate_model.Q
		self.R = candidate_model.R

		self.num_inputs = candidate_model.num_inputs
		assert isinstance(self.num_inputs, int) and self.num_inputs > 0
		self.S_u = candidate_model.S_u

		self.num_states = self.Q.shape[0]
		assert self.num_states == self.H.shape[1]

		self.num_meas = self.H.shape[0]
		assert self.num_meas == self.R.shape[0]

		self.x0   = candidate_model.x0
		self.S_x0 = candidate_model.S_x0

	"""
	Transition function
	"""
	@property
	def f (self):
		return self._f 
	@f.setter
	def f (self, f):
		assert callable(f)
		self._f = f

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
	def Q (self):
		return self._Q
	@Q.setter
	def Q (self, Q):
		assert is_symmetric_matrix(Q)
		self._Q = Q.copy()

	"""
	Measurement noise covariance matrix
	"""
	@property
	def R (self):
		return self._R
	@R.setter
	def R (self, R):
		assert is_symmetric_matrix(R)
		self._R = R.copy()

	"""
	Control input covariance
	"""
	@property
	def S_u (self):
		return self._S_u
	@S_u.setter
	def S_u (self, S_u):
		if S_u is None:
			S_u = np.zeros(( self.num_inputs, self.num_inputs ))
		assert is_symmetric_matrix(S_u)
		assert S_u.shape == (self.num_inputs, self.num_inputs)
		self._S_u = S_u.copy()

	"""
	Initial state mean
	"""
	@property
	def x0 (self):
		return self._x0 
	@x0.setter
	def x0 (self, x0):
		assert x0.shape == (self.num_states,)
		self._x0 = x0.copy()

	"""
	Initial state covariance
	"""
	@property
	def S_x0 (self):
		return self._S_x0 
	@S_x0.setter
	def S_x0 (self, S_x0):
		if S_x0 is None:
			S_x0 = np.zeros(( self.num_states, self.num_states ))
		assert is_symmetric_matrix(S_x0)
		self._S_x0 = S_x0.copy()

	"""
	State constraints
	- For surrogate models that want to enforce state constraints in order
	  not to deviate too far from training data.
	"""
	def initialise_x_constraints (self):
		pass

	def update_x_constraints (self, x, s, dxdU, dsdU):
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
		x0 = x0 if not initial_uncertainty else mvn(x0, self.S_x0)

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
		us = mvn(u, self.S_u)
		xy = self.predict(x, us)
		wk = mvn( np.zeros(self.num_states), self.Q )
		vk = mvn( np.zeros(self.num_meas), self.R )
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

	def predict_y_dist (self, x, s, grad=False):
		"""
		Input: latent state mean x and covariance s
			p( x_k | y_{1 : T} ) = N( x, s )
		Output: observed state mean and variance of observation
			p( y_k | y_{1 : T} ) = N( Y, S )

		If x.ndim == 1, one-step prediction
		If x.ndim == 2, multi-step prediction
		"""
		if x.ndim == 1:
			return self._predict_y_dist(x, s, grad=grad)

		n = len(x)
		Y = np.zeros(( n, self.num_meas ))
		S = np.zeros(( n, self.num_meas, self.num_meas ))
		if not grad:
			for k in range(n):
				Y[k], S[k] = self._predict_y_dist(x[k], s[k], grad=grad)
			return Y, S
		dYdx = np.zeros(( n, self.num_meas, self.num_states ))
		dYds = np.zeros(( n, self.num_meas, self.num_states, self.num_states ))
		dSdx = np.zeros(( n, self.num_meas, self.num_meas, self.num_states ))
		dSds = np.zeros( [n] + [self.num_meas]*2 + [self.num_states]*2 )
		for k in range(n):
			Y[k], S[k], dYdx[k], dYds[k], dSdx[k], dSds[k] \
			                = self._predict_y_dist(x[k], s[k], grad=True)
		return Y, S, dYdx, dYds, dSdx, dSds

	def _predict_y_dist (self, x, s, grad=False):
		Y = np.matmul(self.H, x)
		S = np.matmul(self.H, np.matmul(s, self.H.T) ) + self.R
		if grad:
			dYdm = self.H
			dYds = np.zeros( [self.num_meas] + [self.num_states]*2 )
			dSdx = np.zeros( [self.num_meas]*2 + [self.num_states] )
			dSds = np.zeros( [self.num_meas]*2 + [self.num_states]*2 )
			for e1 in range(  self.num_meas ):
				for e2 in range(  self.num_meas ):
					dSds[e1,e2] = self.H[e1][:,None] * self.H[e2][None,:]
			return Y, S, dYdm, dYds, dSdx, dSds
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
		K  = np.matmul(sH, np.linalg.inv(np.matmul(self.H, sH) + self.R))
		xk = x + np.matmul(K, yk - np.matmul(self.H, x))
		sk = s - np.matmul(K, sH.T)
		return xk, sk

	def predict_filter (self, Y, x0, S_x0, U):
		"""
		Filter sequence, based on observations Y, controls U,
		and with prediction p(x_1) ~ N(x0, S_x0)

		Y    : [ n, E ]       ( y_1, ..., y_n )
		x0   : [ D, ]
		S_x0 : [ D, D ]
		U    : [ n, D_U ]     ( u_1, ..., u_{n-1} )

		Outputs
		x  : [ n+1, D ]       ( x_1, ..., x_n )
		s  : [ n+1, D, D]
		"""
		n = len(Y)
		x = np.zeros(( n, self.num_states ))
		s = np.zeros(( n, self.num_states, self.num_states ))
		M, S = x0, S_x0
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

