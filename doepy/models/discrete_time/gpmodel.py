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

from .model import dtModel
from ..gpmodel import GPModel

from ...utils import assert_not_none
from ...training import generate_training_data
from ...constraints import ConstantMeanStateConstraint
from ...approximate_inference import gp_taylor_moment_match

default_noise_var = 1e-5

class dtGPModel (dtModel, GPModel):
	def __init__ (self, candidate_model, moment_match=gp_taylor_moment_match):
		"""
		We assume we do not have gradient information for f

		if delta_transition:
		    f( x_k, u_k ) = x_k  +  g( x_k, u_k )
		else
		    f( x_k, u_k ) = g( x_k, u_k )

		We place a GP prior on the function g
		"""
		dtModel.__init__(self, candidate_model)
		GPModel.__init__(self, transform=candidate_model.transform)

		assert_not_none(self.u_bounds,'u_bounds')

		assert_not_none(candidate_model.x_bounds,'candidate:x_bounds')
		self.x_bounds      = candidate_model.x_bounds
		self.x_constraints = None

		self.delta_transition = False if candidate_model.delta_transition is None\
                                      else candidate_model.delta_transition

		self.moment_match = moment_match

	"""
	Train GP surrogates
	"""
	def train (self, active_dims=None, noise_var=default_noise_var, hyp=None, **kwargs):
		# Training data dictionary
		dic = {'f':self.f, 'active_dims': active_dims,
			'x_bounds':self.x_bounds, 'u_bounds':self.u_bounds,
			'return_active_dims':True}

		# Model parameters
		if self.num_param > 0:
			assert 'p_bounds' in kwargs, 'Training requires parameter bounds'
			dic['p_bounds'] = kwargs.get('p_bounds')
		# Number of training data points
		nom = 'num_data_points_per_num_dim_combo'
		if nom in kwargs:
			dic[nom] = kwargs.get(nom)
		T, active_dims = generate_training_data(**dic)
		
		# Training targets
		Z = T[-1]
		if self.delta_transition:
			D = np.arange( self.num_states )
			Z = [ z - x[:,d] for z,x,d in zip(Z, T[0], D) ]

		# Training inputs
		T = [ np.concatenate(t, axis=1) for t in zip(*T[:-1]) ]

		self._train(T, Z, active_dims, noise_var, hyp=hyp, **kwargs)


	"""
	State constraints
	"""
	def initialise_x_constraints (self):
		# Find maximum lengthscales
		hyp = np.zeros((self.num_states, self.num_states))
		for e,gp in enumerate(self.gps):
			L = gp.kern.lengthscale
			A = gp.kern.active_dims
			A = A[ A<self.num_states ]
			for a,l in zip(A,L):
				hyp[e,a] = self.input_transform(l, back=True, dim=e)
		hyp = np.max(hyp,axis=0)

		# Find min distances in training data
		Xt = np.vstack([ gp.X for gp in self.gps ])
		Xt = self.input_transform(Xt, back=True)[:,:self.num_states]
		Xt = np.abs(Xt[1:] - Xt[:-1])
		Xt = np.min( np.where(Xt>0, Xt, np.inf), axis=0 )

		# We can allow a little bit of slack in latent states
		slack  = Xt / np.where(hyp>0, hyp, 100.)
		bounds = self.x_bounds + np.c_[-slack, slack]

		self.x_constraints = ConstantMeanStateConstraint(bounds)
		self.c, self.dcdU  = None, None

	def update_x_constraints (self, x, s, dxdU, dsdU, step=None):
		if self.x_constraints is None:
			self.initialise_x_constraints()
		c, dcdx, dcds = self.x_constraints(x, s, grad=True)
		dcdU = np.einsum('ij,njk->ink',dcdx,dxdU) \
			   + np.einsum('ijk,njkd->ind',dcds,dsdU)
		if self.c is None:
			self.c    = c[None,:]
			self.dcdU = dcdU[None,:]
		else:
			self.c    = np.vstack((self.c, c))
			self.dcdU = np.vstack((self.dcdU, dcdU[None,:]))

	def get_x_constraints (self):
		if self.x_constraints is None:
			return None
		i,j,k,l = self.dcdU.shape
		return self.c.reshape((i*j)), self.dcdU.reshape((i*j,k,l))

	def num_x_constraints (self):
		return 2 * self.num_states

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		# Input mean and covariance
		input_mean, input_cov = self.get_input_mean_and_cov(xk, Sk, u, True)

		if self.transform:
			assert_not_none(self.output_transform, 'output_transform')
			assert_not_none(self.input_transform, 'input_transform')
			input_mean = self.input_transform( input_mean )
			input_cov  = self.input_transform.cov( input_cov )

		# Moment matching
		assert not self.gps == [], 'GP surrogate(s) not trained yet.'
		res = self.moment_match(self.gps, input_mean, input_cov, grad=grad)
		if grad:
			M, S, V, do = res
		else:
			M, S, V = res
				
		# Transform back
		if self.transform:
			qt,qz = self.input_transform.q, self.output_transform.q
			M     = self.output_transform(M, back=True)
			S     = self.output_transform.cov(S, back=True)
			qtqz  = qt[:,None] * qz[None,:]
			V    *= qtqz
			if grad:
				qtqt  = qt[:,None] * qt[None,:]
				qzqz  = qz[:,None] * qz[None,:]
				do.dMdm *= qz[:,None] / qt[None,:]
				do.dMds *= qz[:,None,None] / qtqt[None,:,:]
				do.dSdm *= qzqz[:,:,None] / qt[None,None,:]
				do.dSds *= qzqz[:,:,None,None] / qtqt[None,None,:,:]
				do.dVdm *= qtqz[:,:,None] / qt[None,None,:]
				do.dVds *= qtqz[:,:,None,None] / qtqt[None,None,:,:]

		# Separate state and control dimensions again
		V = V[:self.num_states]
		if grad:
			do = self.get_latent_state_derivatives(do)

		# Process noise variance
		S += self.x_covar
		# Delta transition
		if self.delta_transition:
			M += xk
			S += Sk + V + V.T
			V += Sk
			if grad:
				do.dMdx += np.eye(self.num_states)
				do.dSdx += do.dVdx + np.swapaxes(do.dVdx,0,1)
				do.dSds += do.dVds + np.swapaxes(do.dVds,0,1)
				for d1 in range(self.num_states):
					for d2 in range(self.num_states):
						do.dSds[d1,d2,d1,d2] += 1
						do.dVds[d1,d2,d1,d2] += 1

		ret = (M, S, V) if cross_cov else (M, S)
		return ret if not grad else ret+(do,)
