
import numpy as np 
import tensorflow as tf 
#tfco = tf.contrib.constrained_optimization
#ConMinProb = tfco.ConstrainedMinimizationProblem
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance as mvn

#from .. import settings
from gpflow.core.compilable import Build
#from ..params import Parameterized, DataHolder
#from ..decors import autoflow
#from ..mean_functions import Zero

from .training import tf_contrib_constrained_optimization_forked as tfco
ConMinProb = tfco.ConstrainedMinimizationProblem

from gpflow import settings
from gpflow import Param
from gpflow.models import Model
from gpflow.decors import name_scope, params_as_tensors, autoflow
from gpflow.params import Parameterized, ParamList
from gpflow.transforms import Logistic

from .design_criteria import HR

class ProblemInstance (Parameterized, ConMinProb):
	def __init__ (self, models, num_steps, u_bounds, u_delta_lim, y_bounds, 
			y_constraint_prob=0.99, div_criterion=None):
		Parameterized.__init__(self, name='problem')
		ConMinProb.__init__(self)
		self._objective   = None
		self._constraints = None
		#self.pre_train_ops = False

		self.models     = models
		self.num_models = len(models)
		self.num_steps  = num_steps
		self.divergence = div_criterion or HR()

		self.Du = len(u_bounds)
		self.U = ParamList([ Param(np.array([np.mean(ub)]*num_steps), 
							transform=Logistic(ub[0], ub[1]), 
							dtype=settings.float_type) 
						for ub in u_bounds])
		self.u_delta  = u_delta_lim
		self.y_bounds = y_bounds
		self.y_const_prob = y_constraint_prob
		assert isinstance(self.y_const_prob, float) and 0<self.y_const_prob<1

	@property
	def objective (self):
		return self._objective

	@property
	def constraints (self):
		return tf.stack( self._constraints, axis=0 )

	@property
	def likelihood_tensor (self):
		return self._likelihood_tensor

	@autoflow()
	def compute_log_prior (self):
		"""Compute the log prior of the model."""
		return self.prior_tensor

	@autoflow()
	def compute_log_likelihood (self):
		"""Compute the log likelihood of the model."""
		return self.likelihood_tensor

	def is_built (self, graph):
		is_built = super().is_built(graph)
		if is_built is not Build.YES:
			return is_built
		if self._likelihood_tensor is None:
			return Build.NO
		return Build.YES

	def build_objective (self):
		likelihood = self._build_likelihood()
		priors = []
		for param in self.parameters:
			unconstrained = param.unconstrained_tensor
			constrained = param._build_constrained(unconstrained)
			priors.append(param._build_prior(unconstrained, constrained))
		prior = self._build_prior(priors)
		return -tf.add(likelihood, prior, name='objective')

	def _clear (self):
		super(ProblemInstance, self)._clear()
		self._objective = None
		self._constraints = None
		self._likelihood_tensor = None

	def _build (self):
		super(ProblemInstance, self)._build()
		likelihood, constraints = self._build_problem()
		prior = self.prior_tensor
		objective = self._build_objective(likelihood, prior)
		self._likelihood_tensor = likelihood
		self._constraints = constraints
		self._objective = -tf.add(likelihood, prior, name='objective')

	def sample_feed_dict(self, sample):
		tensor_feed_dict = {}
		for param in self.parameters:
			if not param.trainable: continue
			constrained_value = sample[param.pathname]
			unconstrained_value = param.transform.backward(constrained_value)
			tensor = param.unconstrained_tensor
			tensor_feed_dict[tensor] = unconstrained_value
		return tensor_feed_dict

	def _build_objective(self, likelihood_tensor, prior_tensor):
		func = tf.add(likelihood_tensor, prior_tensor, name='nonneg_objective')
		return tf.negative(func, name='objective')

	@name_scope('problem_formulation')
	@params_as_tensors
	def _build_problem (self):
		# Hard inequality constraints
		ineq = self._build_u_delta_constraints()
		# Divergence term
		D  = 0.
		# Means and variances of predicted states and observations 
		mX = [ model.x0.astype(np.float) for model in self.models ]
		sX = [ model.P0.astype(np.float) for model in self.models ]
		mY = [ None for model in self.models ]
		sY = [ None for model in self.models ]
		# Iterate over the control sequence
		for n in range( self.num_steps ):
			# Next control input
			u = tf.concat( [self.U[j][n] for j in range(self.Du)], axis=0 )
			u = tf.reshape( u, [self.Du] )
			# State prediction
			for i, model in enumerate( self.models ):
				mX[i],sX[i] = model._build_predict_x_dist(mX[i],sX[i], u)[:2]
				mY[i],sY[i] = model._build_predict_y_dist(mX[i],sX[i])
				#ineq.append( self._build_state_constraint(mY[i],sY[i]) )
			D += self.divergence( mY, sY )
		return -D, ineq

	@name_scope('likelihood')
	@params_as_tensors
	def _build_likelihood (self):
		return self._build_objective()[0]

	#@name_scope('u_delta_constraints')
	#@params_as_tensors
	def _build_u_delta_constraints (self):
		ineq = []
		# U delta constraints
		for n in range( 1, self.num_steps ):
			for j in range( self.Du ):
				#d_const1 = self.U[j][n] - self.U[j][n-1] + self.u_delta[j]
				#d_const2 = self.U[j][n-1] - self.U[j][n] + self.u_delta[j]
				# | U[n] - U[n-1] | <= u_delta
				"""
				d_const1 = self.U[j][n] - self.U[j][n-1] - self.u_delta[j]
				d_const2 = self.U[j][n-1] - self.U[j][n] - self.u_delta[j]
				ineq.append(d_const1)
				ineq.append(d_const2)
				"""
				d_const = -self.U[j][n]
				ineq.append(d_const)
		return ineq

	#@name_scope('state_constraints')
	#@params_as_tensors
	"""
	def _build_state_constraint (self, mY, mS):
		dist  = mvn(mY, mS)
		Phi_b = dist.cdf(self.y_bounds[:,1])
		Phi_a = dist.cdf(self.y_bounds[:,0])
		return Phi_b - Phi_a - self.y_const_prob
	"""

	@autoflow()
	def control_signal (self):
		return self._build_control_signal()

	@name_scope('control_signal')
	@params_as_tensors
	def _build_control_signal (self):
		U = tf.concat([self.U[j] for j in range(self.Du)], axis=0)
		U = tf.reshape( U, [self.Du, self.num_steps] )
		return tf.transpose(U)

	def sample_feed_dict(self, sample):
		tensor_feed_dict = {}
		for param in self.parameters:
			if not param.trainable: continue
			constrained_value = sample[param.pathname]
			unconstrained_value = param.transform.backward(constrained_value)
			tensor = param.unconstrained_tensor
			tensor_feed_dict[tensor] = unconstrained_value
		return tensor_feed_dict
