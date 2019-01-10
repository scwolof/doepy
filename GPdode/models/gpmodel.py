
from numpy import ndarray, max, min
import tensorflow as tf

from gpflow import settings
from gpflow.params import DataHolder, ParamList
from gpflow.decors import name_scope, params_as_tensors, autoflow
from gpflow.conditionals import base_conditional
from gpflow.logdensities import multivariate_normal

from gpflow.models import Model as GPflowModel
from gpflow.kernels import RBF
from gpflow.training import ScipyOptimizer
from gpflow.transforms import Logistic
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

import logging

from .model import Model
from ..utils import block_diagonal

# Float type
dtype = settings.float_type

class GPModel (GPflowModel, Model):
	def __init__ (self, f, Xtrain, Ytrain, H, Q, R, delta_transition=False, \
			name='name', noise_var=1e-6, noise_trainable=False):
		r"""
		f : transition function x_{k+1} = f(x_k, u_k). We put a GP prior on f.
		Xtrain : N x (D+Du) GP training inputs [ x_k^T, u_k^T ]
		Ytrain : N x D  GP training observations [ x_{k+1}^T ]
		H : observation matrix
		Q : process noise covariance matrix
		R : measurement noise covariance

		name : Name for GPflow model, must be unique
		noise_var : Noise variance for GP model
		noise_trainable : Boolean, optimise noise variance hyperparameter

		Model:
			x_{k+1} = g( x_k, u_k )  +  w_k,   w_k ~ N(0, Q)
				y_k = H * x_k  +  v_k,         v_k ~ N(0, R)
		
		if delta_transition:
			g( x_k, u_k ) = x_k  +  f( x_k, u_k )
		else
			g( x_k, u_k ) = f( x_k, u_k )

		"""
		GPflowModel.__init__(self, name=name)
		Model.__init__(self, f, H, Q, R)

		self.delta_transition = delta_transition
		# Dimensions
		self.num_data   = Xtrain.shape[0]
		self.num_input  = Xtrain.shape[1]
		self.num_latent = Ytrain.shape[1]
		max_lengthscale = 10 * ( max(Xtrain, axis=0) - min(Xtrain, axis=0) )
		self.Du = self.num_input - self.D
		# Training data
		if isinstance(Xtrain, ndarray):
			Xtrain = DataHolder(Xtrain)
		if isinstance(Ytrain, ndarray):
			Ytrain = DataHolder(Ytrain)
		self.X, self.Y = Xtrain, Ytrain
		# Independent covariance functions
		kerns = [ RBF(self.num_input, ARD=True) for _ in range(self.num_latent) ]
		for maxl, kern in zip(max_lengthscale, kerns):
			kern.lengthscales.transform = Logistic(0,maxl)
		self.kerns = ParamList(kerns)
		# Gaussian likelihoods
		likelihoods = [ Gaussian(noise_var) for _ in range(self.num_latent) ]
		for lkh in likelihoods:
			lkh.trainable = noise_trainable
		self.likelihoods = ParamList(likelihoods)
		# Mean functions
		self.mean_functions = Zero(output_dim=1)

	def _build (self):
		# Turn off annoying warning message
		logger = logging.getLogger('gpflow.logdensities')
		logger.setLevel(level=logging.ERROR)
		# Build
		super()._build()
		# Turn warning messages back on again
		logger.setLevel(level=logging.WARNING)

	@name_scope('likelihood')
	@params_as_tensors
	def _build_likelihood (self):
		m = self.mean_functions(self.X)
		logpdf = 0
		for e, kern, lklhood in zip(range(self.num_latent), 
									self.kerns._list, 
									self.likelihoods._list):
			K = kern.K(self.X) + lklhood.variance * \
					tf.eye(self.num_data, dtype=dtype)
			L = tf.cholesky( K )
			t = tf.reshape(self.Y[:,e], [self.num_data,1])
			logpdf = logpdf + tf.reduce_sum( multivariate_normal(t, m, L) )
		return logpdf

	@name_scope('predict')
	@params_as_tensors
	def _build_predict (self, Xnew, full_cov=False):
		m      = self.mean_functions(self.X)
		f_mean = None
		f_var  = None
		for e, kern, lklhood in zip(range(self.num_latent), 
									self.kerns._list, 
									self.likelihoods._list):
			Kmn   = kern.K(self.X, Xnew)
			Kmm   = kern.K(self.X) + lklhood.variance * tf.eye(self.num_data, dtype=dtype)
			Knn   = kern.K(Xnew) if full_cov else kern.Kdiag(Xnew)
			t     = tf.reshape(self.Y[:,e], [self.num_data,1]) - m
			mu,s2 = base_conditional(Kmn, Kmm, Knn, t, full_cov=full_cov, white=False)  
			
			""" Mean """
			mu = mu + self.mean_functions(Xnew)
			if f_mean is None:
				f_mean = mu
			else:
				f_mean = tf.concat([f_mean, mu], 1)
			""" Variance """
			if f_var is None:
				f_var = s2
			else:
				if full_cov:
					f_var = tf.concat([f_var, s2], 0)
				else:
					f_var = tf.concat([f_var, s2], 1)
		return f_mean, f_var

	"""
	Train GP surrogates
	"""
	def optimise (self):
		# Make sure model is compiled
		self.compile()
		# Optimize hyperparameters
		tf.logging.set_verbosity(tf.logging.WARN)
		ScipyOptimizer().minimize(self)
		tf.logging.set_verbosity(tf.logging.INFO)
	# American spelling
	def optimize (self):
		self.optimise()

	"""
	State prediction
	"""
	@name_scope('predict_x')
	@params_as_tensors
	def _build_predict_x_dist (self, xk, Pk, u):
		tnew = tf.concat( [xk, u], axis=0 )
		Pnew = block_diagonal([Pk, tf.zeros((self.Du, self.Du), dtype=dtype)])

		M, S, V = self._build_predict_uncertain_input(tnew, Pnew)

		mu_k = M
		S_k  = S + self.Q
		V_k  = V[:self.D]
		if self.delta_transition:
			mu_k += xk
			S_k  += Pk + V_k + tf.transpose(V_k)
			V_k  += Pk
		return mu_k, S_k, V_k
	
	"""
	Exact moment matching for uncertain input with RBF kernels.
	"""
	@name_scope('predict_uncertain_input')
	@params_as_tensors
	def _build_predict_uncertain_input (self, Xmean, Xvar):
		# Centralise training inputs
		inp = self.X - Xmean

		# Woodbury inverses (K + s2*I)^-1
		iKs = []
		for kern, lklhood in zip(self.kerns._list, self.likelihoods._list):
			Kmm = kern.K(self.X) + lklhood.variance * tf.eye(self.num_data, dtype=dtype)
			Lm  = tf.cholesky( Kmm )
			A   = tf.matrix_triangular_solve(Lm, tf.eye(self.num_data, dtype=dtype), lower=True)
			A   = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)
			iKs.append( A )
		
		# Woodbury vectors (K + s2*I)^-1 (Y - M(X))
		beta = []
		for i, kern in zip(range(self.num_latent), self.kerns._list):
			y = tf.reshape(self.Y[:,i], [self.num_data,1]) - self.mean_functions(self.X)
			A = tf.matmul(iKs[i], y)
			beta.append( A )
			
		logk = [ tf.log(kern.variance) - \
				 0.5 * tf.reduce_sum( tf.div(inp, kern.lengthscales)**2, axis=1 ) \
				for kern in self.kerns._list ]
	
		def mean_covar_terms ():
			for i, kern in enumerate(self.kerns._list):
				lengp = kern.lengthscales
				iL = tf.diag(1. / lengp)
				nn = tf.matmul( inp, iL )
				B  = tf.matmul( iL, tf.matmul(Xvar, iL) ) + tf.eye( self.num_input, dtype=dtype )
				iQ = tf.linalg.inv( B )
				iB = tf.matmul( iL, tf.matmul(iQ, iL) )
				xB = tf.matmul(inp, iB)
				t  = tf.matmul( nn, iQ )
				l  = tf.exp( -0.5 * tf.reduce_sum(nn * t, axis=1) )
				lb = l * beta[i][:,0]
				c  = kern.variance / tf.sqrt(tf.linalg.det(B))
				yield c, lb, xB
		
		M = tf.stack([ c * tf.reduce_sum( lb ) for c,lb,_ in mean_covar_terms() ])
		V = tf.stack([ c * tf.reduce_sum( tf.multiply(lb[:,None], tf.matmul(xB, Xvar)), axis=0 )\
					 for c,lb,xB in mean_covar_terms()], axis=1)
		
		def var_terms (i,j,stop=False):
			kerni = self.kerns._list[i]
			lengi = 1. / tf.square( kerni.lengthscales )
			ii    = tf.multiply(inp, lengi)
			betai = beta[i]

			kernj = self.kerns._list[j]
			lengj = 1. / tf.square( kernj.lengthscales )
			ij    = tf.multiply(inp, lengj)
			betaj = beta[j]

			R    = tf.matmul( Xvar, tf.diag(lengi + lengj) ) + tf.eye( self.num_input, dtype=dtype )
			iRS  = 0.5 * tf.matmul( tf.linalg.inv(R), Xvar )
			Q1   = logk[i][:,None] + logk[j][None,:]
			
			iiR   = tf.matmul(ii, iRS)
			ijR   = tf.matmul(ij, iRS)            
			asum = tf.reduce_sum( iiR*ii, axis=1 )[:,tf.newaxis]
			bsum = tf.reduce_sum( ijR*ij, axis=1 )[tf.newaxis,:]
			ab   = tf.matmul( iiR, tf.transpose(ij) )
			Q2   = asum + bsum + 2*ab
			
			Q    = tf.exp(Q1 + Q2)
			isdR = 1. / tf.sqrt( tf.linalg.det(R) )

			A = tf.einsum('ik,jk->ij', betai, betaj)
			if i == j: # Incorporate model uncertainty
				A -= iKs[i]
			A = tf.multiply(A, Q)

			s = tf.multiply(tf.reduce_sum( A ), isdR)
			if i == j:
				s += kerni.variance
			return s
		
		S  = tf.convert_to_tensor([ [ var_terms(i,j) for j in range(self.num_latent) ] \
								   for i in range(self.num_latent) ], dtype=dtype)
		S -= tf.multiply(M[:,None], M[None,:])
		return M, S, V