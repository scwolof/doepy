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

from GPy.models import GPRegression
from GPy.kern import RBF

import logging
logging.getLogger('GP').propagate = False

from .core_model import CoreModel

from ..training import generate_training_data
from ..transform import BoxTransform, MeanTransform
from ..constraints import MeanStateConstraint
from ..approximate_inference import rbf_moment_match

class GPModel (CoreModel):
	def __init__ (self, f, num_inputs, *args, x_bounds, delta_transition=False,\
		          transform=True, **kwargs):
		"""
		We assume we do not have gradient information for f

		if delta_transition:
			f( x_k, u_k ) = x_k  +  g( x_k, u_k )
		else
			f( x_k, u_k ) = g( x_k, u_k )

		We place a GP prior on the function g

		transform : transform [x,u] -> [0, 1]^dim, and ( x_{k+1} - m) / std
		## WARNING - transform suffering problems ##
		"""
		super().__init__(f, num_inputs, *args, **kwargs)

		self.gps = []
		self.hyp = []
		self.x_bounds     = x_bounds
		self.x_constraint = None
		self.transform    = transform
		if self.transform:
			self.z_transform  = None
			self.t_transform  = None
		self.delta_transition = delta_transition

	"""
	Train GP surrogates
	"""
	def _gp_regression (self, X, Y, kern, **kwargs):
		return GPRegression(X, Y, kern)

	def train (self, X, U, Z=None, hyp=None, noise_var=1e-6, **kwargs):
		assert len(U) == len(X)
		
		if Z is None:
			if self.delta_transition:
				Z = np.array([ self.f(x,u) - x for x,u in zip(X,U) ])
			else:
				Z = np.array([ self.f(x,u) for x,u in zip(X,U) ])
		Tt, Zt = self._training_data(np.c_[ X, U ], Z)

		self.hyp = []
		for d in range(self.num_states):
			dim  = Tt.shape[1]
			kern = RBF(input_dim=dim, ARD=True)
			gp   = self._gp_regression(Tt, Zt[:,[d]], kern, **kwargs)
			if hyp is None:
				# Constrain noise variance
				gp.Gaussian_noise.variance.constrain_fixed(noise_var)
				# Constrain lengthscales
				LS = np.max(Tt, axis=0) - np.min(Tt, axis=0)
				for dd in range(dim):
					gp.kern.lengthscale[[dd]].constrain_bounded(
						lower=0., upper=10.*LS[dd], warning=False )
				gp.optimize()
			else:
				gp.update_model(False)
				gp.initialize_parameter()
				gp[:] = hyp[d]
				gp.update_model(True)
			self.hyp.append( gp[:] )
			self.gps.append( gp )

	"""
	Transform training data
	"""
	def _training_data (self, T, Z):
		if not self.transform:
			return T, Z
		self.z_transform = MeanTransform( Z )
		self.t_transform = BoxTransform( T )
		return self.t_transform(T), self.z_transform(Z)

	"""
	State constraints
	"""
	def initialise_x_constraint (self):
		self.x_constraint = MeanStateConstraint(self.x_bounds)
		self.c, self.dcdU = None, None

	def update_x_constraint (self, x, p, dxdU, dpdU):
		if self.x_constraint is None:
			self.initialise_x_constraint()
		c, dcdx, dcdp = self.x_constraint(x, p, grad=True)
		dcdU = np.einsum('ij,jnk->ink',dcdx,dxdU) \
		       + np.einsum('ijk,jknd->ind',dcdp,dpdU)
		if self.c is None:
			self.c    = c[None,:]
			self.dcdU = dcdU[None,:]
		else:
			self.c    = np.vstack((self.c, c))
			self.dcdU = np.vstack((self.dcdU, dcdU))

	def get_x_constraint (self):
		if self.x_constraint is None:
			return None
		return self.c, self.dcdU

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		if self.transform:
			assert self.z_transform is not None
			assert self.t_transform is not None
		assert not self.gps == [], 'GP surrogate(s) not trained yet.'

		# Input mean and variance
		tnew = np.array( xk.tolist() + u.tolist() )
		dim  = len( tnew )
		Snew = np.zeros((dim, dim))
		Snew[:self.num_states, :self.num_states] = Sk
		Snew[self.num_states:, self.num_states:] = self.Su
		if self.transform:
			tnew = self.t_transform(tnew)
			Snew = self.t_transform.cov(Snew)

		# Moment matching
		res = rbf_moment_match(self.gps, tnew, Snew, grad=grad)
		if grad:
			M, S, V, dMdt, dMds, dSdt, dSds, dVdt, dVds = res
		else:
			M, S, V = res
				
		# Transform back
		if self.transform:
			qt,qz = self.t_transform.q, self.z_transform.q
			M     = self.z_transform(M, back=True)
			S     = self.z_transform.cov(S, back=True)
			qtqz  = qt[:,None] * qz[None,:]
			V    *= qtqz
			if grad:
				qtqt  = qt[:,None] * qt[None,:]
				qzqz  = qz[:,None] * qz[None,:]
				dMdt *= qz[:,None] / qt[None,:]
				dMds *= qz[:,None,None] / qtqt[None,:,:]
				dSdt *= qzqz[:,:,None] / qt[None,None,:]
				dSds *= qzqz[:,:,None,None] / qtqt[None,None,:,:]
				dVdt *= qtqz[:,:,None] / qt[None,None,:]
				dVds *= qtqz[:,:,None,None] / qtqt[None,None,:,:]

		# Separate state and control dimensions again
		V = V[:self.num_states]
		if grad:
			dMdx = dMdt[:,:self.num_states]
			dMdu = dMdt[:,self.num_states:]
			dMds = dMds[:,:self.num_states,:self.num_states]
			dSdx = dSdt[:,:,:self.num_states]
			dSdu = dSdt[:,:,self.num_states:]
			dSds = dSds[:,:,:self.num_states,:self.num_states]
			dVdx = dVdt[:self.num_states,:,:self.num_states]
			dVdu = dVdt[:self.num_states,:,self.num_states:]
			dVds = dVds[:self.num_states,:,:self.num_states,:self.num_states]

		# Process noise variance
		S += self.Q
		# Delta transition
		if self.delta_transition:
			M += xk
			S += Sk + V + V.T
			V += Sk
			if grad:
				dMdx += np.eye(self.num_states)
				dSdx += dVdx + np.swapaxes(dVdx,0,1)
				dSds += dVds + np.swapaxes(dVds,0,1)
				for d1 in range(self.num_states):
					for d2 in range(self.num_states):
						dSds[d1,d2,d1,d2] += 1
						dVds[d1,d2,d1,d2] += 1
		# Returns
		if not grad:
			return (M, S, V) if cross_cov else (M, S)
		if not cross_cov:
			return M, S, dMdx, dMds, dMdu, dSdx, dSds, dSdu
		return M, S, V, dMdx, dMds, dMdu, dSdx, dSds, dSdu
