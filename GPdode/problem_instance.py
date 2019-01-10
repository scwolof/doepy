
import numpy as np 
import tensorflow as tf 

from gpflow import settings
from gpflow import Param
from gpflow.models import Model
from gpflow.decors import name_scope, params_as_tensors, autoflow
from gpflow.params import Parameterized, ParamList
from gpflow.transforms import Logistic

from .design_criteria import HR

class ProblemInstance (Model):
	def __init__ (self, models, num_steps, u_bounds, div_criterion=None):
		Model.__init__(self, name='problem')
		self.models     = models
		self.num_models = len(models)
		self.num_steps  = num_steps
		self.divergence = div_criterion or HR()

		self.Du = len(u_bounds)
		self.U = ParamList([ Param(np.array([np.mean(ub)]*num_steps), 
							transform=Logistic(ub[0], ub[1]), 
							dtype=settings.float_type) 
						for ub in u_bounds])

	@name_scope('likelihood')
	@params_as_tensors
	def _build_likelihood (self):
		# Divergence term
		D  = 0.
		# Constraint violation term
		#C  = 0.
		# Means and variances of predicted states and observations 
		mX = [ model.x0.astype(np.float) for model in self.models ]
		sX = [ model.P0.astype(np.float) for model in self.models ]
		mY = [ None for model in self.models ]
		sY = [ None for model in self.models ]
		# Iterate over the control sequence
		for n in range( self.num_steps ):
			u = tf.concat( [self.U[j][n] for j in range(self.Du)], axis=0 )
			u = tf.reshape( u, [self.Du] )
			for i, model in enumerate( self.models ):
				mX[i],sX[i] = model._build_predict_x_dist(mX[i],sX[i], u)[:2]
				mY[i],sY[i] = model._build_predict_y_dist(mX[i],sX[i])
			D += self.divergence( mY, sY )
		#return C - D
		return -D

	@autoflow()
	def get_U (self):
		return self._build_U()

	@name_scope('control_sequence')
	@params_as_tensors
	def _build_U (self):
		U = tf.concat([self.U[j] for j in range(self.Du)], axis=0)
		U = tf.reshape( U, [self.Du, self.num_steps] )
		return tf.transpose(U)



