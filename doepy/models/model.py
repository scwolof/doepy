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
	def __init__ (self, f, num_inputs, H, Q, R, x0, P0=None, Su=None):
		"""
		f  : transition function x_{k+1} = f(x_k, u_k)
		H  : observation matrix
		Q  : process noise covariance matrix
		R  : measurement noise covariance
		x0 : Initial state mean
		P0 : Initial state covariance
		Su : control input covariance

		Model:
			x_{k+1} = f( x_k, u_k )  +  w_k,   w_k ~ N(0, Q)
			    y_k = H * x_k  +  v_k,         v_k ~ N(0, R)
		with 
			x_0 ~ N(x0, P0), u_k ~ N(u_k, Su)
			u_k of dimension num_inputs
		"""
		self.f  = f
		self.H  = H
		self.Q  = Q
		self.R  = R

		self.num_inputs = num_inputs
		assert isinstance(self.num_inputs, int) and self.num_inputs > 0
		self.Su = Su

		self.num_states = self.Q.shape[0]
		assert self.num_states == self.H.shape[1]

		self.num_meas = self.H.shape[0]
		assert self.num_meas == self.R.shape[0]

		self.x0 = x0
		self.P0 = P0

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
	def Su (self):
		return self._Su
	@Su.setter
	def Su (self, Su):
		if Su is None:
			Su = np.zeros(( self.num_inputs, self.num_inputs ))
		assert is_symmetric_matrix(Su)
		assert Su.shape == (self.num_inputs, self.num_inputs)
		self._Su = Su.copy()

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
	def P0 (self):
		return self._P0 
	@P0.setter
	def P0 (self, P0):
		if P0 is None:
			P0 = np.zeros(( self.num_states, self.num_states ))
		assert is_symmetric_matrix(P0)
		self._P0 = P0.copy()

	"""
	State constraints
	- For surrogate models that want to enforce state constraints in order
	  not to deviate too far from training data.
	"""
	def initialise_x_constraint (self):
		pass

	def update_x_constraint (self, x, p, dxdU, dpdU):
		pass

	def get_x_constraint (self):
		return None


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
		x0 = x0 if not initial_uncertainty else mvn(x0, self.P0)

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
		us = mvn(u, self.Su)
		xy = self.predict(x, us)
		wk = mvn( np.zeros(self.num_states), self.Q )
		vk = mvn( np.zeros(self.num_meas), self.R )
		return xy[0] + wk, xy[1] + vk

	def predict_x_dist (self, xk, Sk, U, cross_cov=False, grad=False):
		"""
		Input state posterior mean xk and variance Pk, and controls U 
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

	def _predict_x_dist (self, xk, Pk, u, cross_cov=False, grad=False):
		# Implemented in children classes (e.g. LinearModel)
		raise NotImplementedError

	def predict_y_dist (self, m, S, grad=False):
		"""
		Input state mean m and variance S
			p( x_k | y_{1 : T} ) = N( m, S )
		Outputs mean and variance of observation
			p( y_k | y_{1 : T} ) = N( mu, s2 )

		If m.ndim == 1, one-step prediction
		If m.ndim == 2, multi-step prediction
		"""
		if m.ndim == 1:
			return self._predict_y_dist(m, S, grad=grad)

		n = len(m)
		Y = np.zeros(( n, self.num_meas ))
		P = np.zeros(( n, self.num_meas, self.num_meas ))
		if not grad:
			for k in range(n):
				Y[k], P[k] = self._predict_y_dist(m[k], S[k], grad=grad)
			return Y, P
		dYdm = np.zeros(( n, self.num_meas, self.num_states ))
		dYds = np.zeros(( n, self.num_meas, self.num_states, self.num_states ))
		dPdm = np.zeros(( n, self.num_meas, self.num_meas, self.num_states ))
		dPds = np.zeros( [n] + [self.num_meas]*2 + [self.num_states]*2 )
		for k in range(n):
			Y[k], P[k], dYdm[k], dYds[k], dPdm[k], dPds[k] \
			                = self._predict_y_dist(m[k], S[k], grad=True)
		return Y, P, dYdm, dYds, dPdm, dPds

	def _predict_y_dist (self, m, S, grad=False):
		mu = np.matmul(self.H, m)
		s2 = np.matmul(self.H, np.matmul(S, self.H.T) ) + self.R
		if grad:
			dmudm  = self.H
			dmudS  = np.zeros( [self.num_meas] + [self.num_states]*2 )
			ds2dmu = np.zeros( [self.num_meas]*2 + [self.num_states] )
			ds2dS  = np.zeros( [self.num_meas]*2 + [self.num_states]*2 )
			for e1 in range(  self.num_meas ):
				for e2 in range(  self.num_meas ):
					ds2dS[e1,e2] = self.H[e1][:,None] * self.H[e2][None,:]
			return mu, s2, dmudm, dmudS, ds2dmu, ds2dS
		return mu, s2

	def filter (self, yk, m, S):
		"""
		Input observation yk, prediction mean m and variance S 
			p( x_k | y_{1 : k-1} ) = N( m, S )
		Outputs state posterior mean mk and variance Pk 
			p( x_k | y_{1 : k} ) = N( mk, Pk )

		yk : [ (n), E ]
		m  : [ (n), D ]
		S  : [ (n), D, D ]
		"""
		assert yk.ndim == m.ndim

		if yk.ndim == 1:
			return self._filter(Y, m, S)

		n = len(yk)
		X = np.zeros(( n, self.num_states ))
		P = np.zeros(( n, self.num_states, self.num_states ))
		for k in range(n):
			X[k], P[k] = self._filter(yk[k], m[k], S[k])
		return X, P

	def _filter (self, yk, m, S):
		SH = np.matmul(S, self.H.T)
		K  = np.matmul(SH, np.linalg.inv(np.matmul(self.H, SH) + self.R))
		mk = m + np.matmul(K, yk - np.matmul(self.H, m))
		Pk = S - np.matmul(K, SH.T)
		return mk, Pk

	def predict_filter (self, Y, x0, P0, U):
		"""
		Filter sequence, based on observations Y, controls U,
		and with prediction p(x_1) ~ N(x0, P0)

		Y  : [ n, E ]		( y_1, ..., y_n )
		x0 : [ D, ]
		P0 : [ D, D ]
		U  : [ n, D_U ]		( u_1, ..., u_{n-1} )

		Outputs
		X  : [ n+1, D ]     ( x_1, ..., x_n )
		P  : [ n+1, D, D]
		"""
		n = len(Y)
		X = np.zeros(( n, self.num_states ))
		P = np.zeros(( n, self.num_states, self.num_states ))
		m, S = x0, P0
		for k in range( n ):
			X[k], P[k] = self._filter(Y[k], m, S)
			if k < n-1:
				m, S = self.predict_x_dist(X[k], P[k], U[k])
		return X, P

	def smooth (self, X, P, U):
		"""
		Iteratively smooths sequences X = [x_1, ..., x_n] and 
		P = [P_1, ..., P_n] with controls U = [u_1, ..., u_n]
		"""
		n  = len(X)
		Xs = np.zeros(( n, self.num_states ))
		Ps = np.zeros(( n, self.num_states, self.num_states ))
		Xs[-1] = X[-1]
		Ps[-1] = P[-1]
		for k in np.arange(1, n)[::-1]:
		    Xs[k-1],Ps[k-1] = self._smooth(X[k-1], P[k-1], Xs[k], Ps[k], U[k-1])
		return Xs, Ps

	def _smooth (self, xk, Pk, xk1, Pk1, u_k):
		"""
		Inputs xk, Pk, xk1, P_{k+1}, u_k
			p( x_k | y_{1:k} ) = N( xk, Pk )
			p( x_{k+1} | y_{1:T}, u_k ) = N( xk1, Pk1 )

		Produces smoothed x_k, P_k
			p( x_k | y_{1:T} ) = N( xs, Ps )
		"""
		m, S, V = self.predict_x_dist(xk, Pk, u_k, cross_cov=True)
		J  = np.matmul( V, np.linalg.inv(S) )
		xs = xk + np.matmul( J, xk1 - m )
		Ps = Pk + np.matmul( J, np.matmul( Pk1 - S, J.T ) )
		return xs, Ps

