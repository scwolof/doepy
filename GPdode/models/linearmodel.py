
import numpy as np 
import tensorflow as tf

from gpflow import settings
from gpflow.params import Parameterized
from gpflow.decors import name_scope, params_as_tensors, autoflow

from .model import Model
from ..utils import block_diagonal

# Float type
dtype = settings.float_type

class LinearModel (Parameterized, Model):
	def __init__ (self, F, B, H, Q, R, name='name'):
		r"""
		F : state transition matrix
		B : control matrix
		H : observation matrix
		Q : process noise covariance matrix
		R : measurement noise covariance

		Model:
			x_{k+1} = F * x_k  +  B * u_k  +  w_k,   w_k ~ N(0, Q)
			    y_k = H * x_k  +  v_k,               v_k ~ N(0, R)
		"""
		Parameterized.__init__(self, name=name)
		f = lambda x,u: np.matmul(F,x) + np.matmul(B,u)
		Model.__init__(self, f, H, Q, R)

		self.F  = F
		self.B  = B
		self.Du = self.B.shape[1]

	def _build (self):
		# Turn off annoying warning message
		logger = logging.getLogger('gpflow.logdensities')
		logger.setLevel(level=logging.ERROR)
		# Build
		super()._build()
		# Turn warning messages back on again
		logger.setLevel(level=logging.WARNING)

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
		self.tfF = tf.constant(F.tolist(), dtype=settings.float_type)
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
		self.tfB = tf.constant(B.tolist(), dtype=settings.float_type)
	@B.deleter
	def B (self):
		del self._B

	"""
	Train model
	"""
	def optimise (self):
		print('No hyperparameters to train.')
	# American spelling
	def optimize (self):
		self.optimise()

	"""
	State prediction
	"""
	@name_scope('predict_x')
	@params_as_tensors
	def _build_predict_x_dist (self, xk, Pk, u):
		xk = tf.reshape(xk, [self.D, 1])
		u  = tf.reshape(u, [self.Du, 1])
		mk = tf.matmul(self.tfF, xk) + tf.matmul(self.tfB, u)
		mk = tf.reshape(mk, [self.D])
		Vk = tf.matmul(Pk, self.tfF, transpose_b=True)
		Sk = tf.matmul(self.tfF, Vk) + self.Q
		return mk, Sk, Vk
