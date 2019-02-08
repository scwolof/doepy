
import numpy as np 

from .model import Model

class LinearModel (Model):
	def __init__ (self, F, B, H, Q, R):
		"""
		F : state transition matrix
		B : control matrix
		H : observation matrix
		Q : process noise covariance matrix
		R : measurement noise covariance

		Model:
			x_{k+1} = F * x_k  +  B * u_k  +  w_k,   w_k ~ N(0, Q)
				y_k = H * x_k  +  v_k,               v_k ~ N(0, R)
		"""
		f = lambda x,u: np.matmul(F,x) + np.matmul(B,u)
		super(LinearModel, self).__init__(f, H, Q, R, B.shape[1])

		self.F  = F
		self.B  = B

	"""
	Transition matrix
	"""
	@property
	def F (self):
		return self._F
	@F.setter
	def F (self, F):
		assert isinstance(F, np.ndarray)
		assert F.shape == (self.num_states, self.num_states)
		self._F  = F.copy()
	@F.deleter
	def F (self):
		del self._F

	"""
	Control input matrix
	"""
	@property
	def B (self):
		return self._B
	@B.setter
	def B (self, B):
		assert isinstance(B, np.ndarray)
		assert B.shape[0] == self.num_states
		self._B  = B.copy()
	@B.deleter
	def B (self):
		del self._B

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		M = np.matmul(self.F, xk) + np.matmul(self.B, u)
		V = np.matmul(Sk, self.F.T)
		S = np.matmul(self.F, V) + self.Q
		if not grad:
			return (M, S, V) if cross_cov else (M, S)
		# Compute gradients
		dMdx = self.F.copy()
		dMds = np.zeros([self.num_states]*3)
		dMdu = self.B.copy()
		dSdx = np.zeros([self.num_states]*3)
		dSds = np.zeros([self.num_states]*4)
		for d1 in range( self.num_states ):
			for d2 in range( self.num_states ):
				dSds[d1,d2] = self.F[d1][:,None] * self.F[d2][None,:]
		dSdu = np.zeros(( self.num_states, self.num_states, self.num_inputs ))
		if not cross_cov:
			return M, S, dMdx, dMds, dMdu, dSdx, dSds, dSdu
		# Compute cross-covariance
		dVdx = np.zeros([self.num_states]*3)
		dVds = np.zeros([self.num_states]*4)
		for d1 in range( self.num_states ):
			dVds[d1,:,d1] = self.F.copy()
		dVdu = np.zeros(( self.num_states, self.num_states, self.num_inputs ))
		return M, S, V, dMdx, dMds, dMdu, dSdx, dSds, dSdu, dVdx, dVds, dVdu
