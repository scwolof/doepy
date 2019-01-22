
import numpy as np 
from numpy.random import multivariate_normal as mvn

from md_code.utils import is_symmetric_matrix, is_pos_def

class Model:
	def __init__ (self, f, H, Q, R):
		"""
		f : transition function x_{k+1} = f(x_k, u_k)
		H : observation matrix
		Q : process noise covariance matrix
		R : measurement noise covariance

		Model:
			x_{k+1} = f( x_k, u_k )  +  w_k,   w_k ~ N(0, Q)
			    y_k = H * x_k  +  v_k,         v_k ~ N(0, R)
		"""
		self.f  = f
		self.H  = H
		self.Q  = Q
		self.R  = R

		self.D = self.Q.shape[0]
		assert self.D == self.H.shape[1]

		self.E = self.H.shape[0]
		assert self.E == self.R.shape[0]

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
	@f.deleter
	def f (self):
		del self._f	

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
		self._H = H
	@H.deleter
	def H (self):
		del self._H

	"""
	Process noise covariance matrix
	"""
	@property
	def Q (self):
		return self._Q
	@Q.setter
	def Q (self, Q):
		assert is_symmetric_matrix(Q)
		self._Q = Q
	@Q.deleter
	def Q (self):
		del self._Q

	"""
	Measurement noise covariance matrix
	"""
	@property
	def R (self):
		return self._R
	@R.setter
	def R (self, R):
		assert is_symmetric_matrix(R)
		self._R = R
	@R.deleter
	def R (self):
		del self._R

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
		X = np.zeros(( n+1, self.D ))
		Y = np.zeros((   n, self.E ))

		X[0] = x0
		for k in range(n):
			X[k+1], Y[k] = self._predict(X[k], U[k])
		return X, Y

	def _predict (self, x, u):
		xk1 = self.f(x, u)
		yk  = np.matmul(self.H, x)
		return xk1, yk

	def sample (self, x0, U):
		"""
		Stochastic model simulation
			x_{k+1} = f( x_k, u_k ) + w_k
			    y_k = H * x_k + v_k

		If U.ndim == 1, one-step prediction
		If U.ndim == 2, multi-step prediction
		"""
		if U.ndim == 1:
			return self._sample(x0, U)

		n = len(U)
		X = np.zeros(( n+1, self.D ))
		Y = np.zeros((   n, self.E ))

		X[0] = x0
		for k in range(n):
			X[k+1], Y[k] = self._sample(X[k], U[k])
		return X, Y

	def _sample (self, x, u):
		xk1, yk = self.predict(x, u)
		wk = mvn( np.zeros(self.D), self.Q )
		vk = mvn( np.zeros(self.E), self.R )
		return xk1+wk, yk+vk

	def predict_x_dist (self, xk, Pk, U, cross_cov=False):
		"""
		Input state posterior mean xk and variance Pk, and controls U 
			p( x_k | y_{1 : k} ) = N( xk, Pk )
		Outputs mean and variance of state prediction
			p( x_{k+1} | y_{1 : k} ) = int[ f(x, u) * N(x | xk, Pk) ] dx

		If U.ndim == 1, one-step prediction.
		If U.ndim == 2, multi-step prediction
		"""
		if U.ndim == 1:
			return self._predict_x_dist(xk, Pk, U, cross_cov=cross_cov)

		n = len(U)
		X = np.zeros(( n+1, self.D ))
		P = np.zeros(( n+1, self.D, self.D ))

		X[0] = xk
		P[0] = Pk
		for k in range(n):
			X[k+1], P[k+1] = self._predict_x_dist(X[k], P[k], U[k])
		return X, P

	def _predict_x_dist (self, xk, Pk, u, cross_cov=False):
		raise NotImplementedError

	def predict_y_dist (self, m, S):
		"""
		Input state mean m and variance S
			p( x_k | y_{1 : T} ) = N( m, S )
		Outputs mean and variance of observation
			p( y_k | y_{1 : T} ) = N( mu, s2 )

		If m.ndim == 1, one-step prediction
		If m.ndim == 2, multi-step prediction
		"""
		if m.ndim == 1:
			return self._predict_y_dist(m, S)

		n = len(m)
		Y = np.zeros(( n, self.E ))
		P = np.zeros(( n, self.E, self.E ))
		for k in range(n):
			Y[k], P[k] = self._predict_y_dist(m[k], S[k])
		return Y, P

	def _predict_y_dist (self, m, S):
		mu = np.matmul(self.H, m)
		s2 = np.matmul(self.H, np.matmul(S, self.H.T) ) + self.R
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
		X = np.zeros(( n, self.D ))
		P = np.zeros(( n, self.D, self.D ))
		for k in range(n):
			X[k], P[k] = self._filter(yk[k], m[k], S[k])
		return X, P

	def _filter (self, yk, m, S):
		SH   = np.matmul(S, self.H.T)
		K    = np.matmul(SH, np.linalg.inv(np.matmul(self.H, SH) + self.R))
		mk   = m + np.matmul(K, yk - np.matmul(self.H, m))
		Pk   = S - np.matmul(K, SH.T)
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
		X = np.zeros(( n, self.D ))
		P = np.zeros(( n, self.D, self.D ))
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
		Xs = np.zeros(( n, self.D ))
		Ps = np.zeros(( n, self.D, self.D ))
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

