
import numpy as np
from numpy.random import multivariate_normal as mvn
import tensorflow as tf

from gpflow import settings
from gpflow.decors import name_scope, params_as_tensors, autoflow

from ..utils import is_symmetric_matrix, is_pos_def


class Model:
	def __init__ (self, f, H, Q, R):
		self.D = Q.shape[0]
		assert self.D == H.shape[1]

		self.E = H.shape[0]
		assert self.E == R.shape[0]
		
		self.f = f
		self.H = H.copy()
		self.Q = Q.copy()
		self.R = R.copy()

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
		self._H  = H
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
		self._Q  = Q
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
		self._R  = R
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
		X = zeros(( n+1, self.D ))
		Y = zeros((   n, self.E ))

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
		xk1 += mvn( np.zeros(self.D), self.Q )
		yk  += mvn( np.zeros(self.E), self.R )
		return xk1, yk
	




class FlowModel (Model):
	def __init__ (self, f, H, Q, R):
		Model.__init__(self, f, H, Q, R)
		self.tfH = tf.constant(H.tolist(), dtype=settings.float_type)
		self.tfQ = tf.constant(Q.tolist(), dtype=settings.float_type)
		self.tfR = tf.constant(R.tolist(), dtype=settings.float_type)

	@autoflow((settings.float_type, [None, None]))
	def predict_f (self, Xnew):
		"""
		Compute the mean and variance of the latent function(s) at the points
		Xnew.
		"""
		return self._build_predict(Xnew)

	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]))
	def predict_f_uncertain_input (self, Xmean, Xvar):
		"""
		Compute the mean and variance of the latent function(s) at the 
		uncertain point Xnew \sim N(Xmean, Xvar)
		"""
		return self._build_predict_uncertain_input( Xmean, Xvar )

	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]),\
			(settings.float_type, [None]))
	def predict_x_dist (self, xk, Pk, u):
		"""
		Input state posterior mean xk and variance Pk, and controls U 
			p( x_k | y_{1 : k} ) = N( xk, Pk )
		Outputs mean and variance of state prediction
			p( x_{k+1} | y_{1 : k}, u ) = int[ f(x, u) * N(x | xk, Pk) ] dx
		and cross-covariance cov[x_k, x_{k+1}]
		"""
		return self._build_predict_x_dist(xk, Pk, u)[:2]
	
	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]),\
			(settings.float_type, [None]))
	def predict_x_dist_covar (self, xk, Pk, u):
		"""
		Input state posterior mean xk and variance Pk, and controls U 
			p( x_k | y_{1 : k} ) = N( xk, Pk )
		Outputs mean and variance of state prediction
			p( x_{k+1} | y_{1 : k}, u ) = int[ f(x, u) * N(x | xk, Pk) ] dx
		and cross-covariance cov[x_k, x_{k+1}]
		"""
		return self._build_predict_x_dist(xk, Pk, u)

	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]))
	def predict_y_dist (self, m, S):
		"""
		Input state mean m and variance S
			p( x_k | y_{1 : T} ) = N( m, S )
		Outputs mean and variance of observation
			p( y_k | y_{1 : T} ) = N( mu, s2 )
		"""
		return self._build_predict_y_dist(m, S)

	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]),\
			(settings.float_type, [None]))
	def predict_dist (self, xk, Pk, u):
		m_x, S_x = self._build_predict_x_dist(xk, Pk, u)[:2]
		m_y, S_y = self._build_predict_y_dist(xk, Pk)
		return m_x, S_x, m_y, S_y

	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]),\
			(settings.float_type, [None]), (settings.int_type, []))
	def predict_dist_iterative (self, xk, Pk, U, num_steps):
		for i in tf.range( num_steps ):
			m_x, S_x = self._build_predict_x_dist(xk, Pk, u)[:2]
			m_y, S_y = self._build_predict_y_dist(xk, Pk)
		return m_x, S_x, m_y, S_y
	
	@autoflow((settings.float_type, [None]), (settings.float_type, [None]),\
			(settings.float_type, [None, None]))
	def filter_x_dist (self, yk, m, S):
		"""
		Input observation yk, prediction mean m and variance S 
			p( x_k | y_{1 : k-1} ) = N( m, S )
		Outputs state posterior mean mk and variance Pk 
			p( x_k | y_{1 : k} ) = N( mk, Pk )

		yk : [ (n), E ]
		m  : [ (n), D ]
		S  : [ (n), D, D ]
		"""
		return self._build_filter_x_dist(yk, m, S)

	@autoflow((settings.float_type, [None]), (settings.float_type, [None, None]),\
			  (settings.float_type, [None]), (settings.float_type, [None, None]),\
			  (settings.float_type, [None]))
	def smooth_x_dist (self, x1, P1, x2, P2, u):
		"""
		Smoothened posterior
			p(x_k | y_{1 : T}, u_{1 : T}) = N( xs_k, Ps_k )
		where
			p( x_k     | y_{1 : k}, u_{1 : k} ) = N(x1, P1)
			p( x_{k+1} | y_{1 : T}, u_{1 : T} ) = N(x2, P2)
			p( u_k ) = N(U1, S_U)
		"""
		return self._build_smooth_x(x1, P1, x2, P2, u)

	@name_scope('predict_observation')
	@params_as_tensors
	def _build_predict_y_dist (self, m, S):
		mu = tf.matmul(self.tfH, tf.reshape(m, [self.D, 1]))
		mu = tf.reshape(mu, [self.E])
		s2 = tf.matmul(self.tfH, tf.matmul(S, self.tfH, transpose_b=True) ) + self.tfR
		return mu, s2

	@name_scope('filter_state')
	@params_as_tensors
	def _build_filter_x_dist (self, yk, m, S):
		m  = tf.reshape(m, [self.D, 1])
		yk = tf.reshape(yk, [self.E, 1])
		SH = tf.matmul(S, self.tfH, transpose_b=True)
		K  = tf.matmul(SH, tf.linalg.inv( tf.matmul(self.tfH, SH) + self.tfR ) )
		tmp = tf.matmul(self.tfH, m)
		mk = m + tf.matmul(K, yk - tmp)
		Pk = S - tf.matmul(K, SH, transpose_b=True)
		mk = tf.reshape(mk, [self.D])
		return mk, Pk

	@name_scope('smooth_state')
	@params_as_tensors
	def _build_smooth_x (self, x1, P1, x2, P2, u):
		"""
		Inputs xk, Pk, xk1, P_{k+1}, u_k
			p( x_k | y_{1:k} ) = N( xk, Pk )
			p( x_{k+1} | y_{1:T}, u_k ) = N( xk1, Pk1 )

		Produces smoothed x_k, P_k
			p( x_k | y_{1:T} ) = N( xs, Ps )
		"""
		m, S, V = self._build_predict_x_dist(x1, P1, u)
		x1 = tf.reshape(x1, [self.D, 1])
		J  = tf.matmul( V, tf.linalg.inv(S) )
		xs = x1 + tf.matmul( J, tf.reshape(x2 - m, [self.D, 1]) )
		Ps = P1 + tf.matmul( J, tf.matmul( P2 - S, J, transpose_b=True ) )
		xs = tf.reshape(xs, [self.D])
		return xs, Ps
