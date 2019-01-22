
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
		super(LinearModel, self).__init__(f, H, Q, R)

		self.F  = F
		self.B  = B
		self.Du = self.B.shape[1]

	"""
	Transition matrix
	"""
	@property
	def F (self):
		return self._F
	@F.setter
	def F (self, F):
		assert isinstance(F, np.ndarray)
		assert F.shape == (self.D, self.D)
		self._F  = F
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
		assert B.shape[0] == self.D
		self._B  = B
	@B.deleter
	def B (self):
		del self._B

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Pk, u, cross_cov=False):
		mk = np.matmul(self.F, xk) + np.matmul(self.B, u)
		Vk = np.matmul(Pk, self.F.T)
		Sk = np.matmul(self.F, Vk) + self.Q
		return (mk, Sk, Vk) if cross_cov else (mk, Sk)
