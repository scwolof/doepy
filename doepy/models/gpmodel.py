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

from os.path import isfile
import numpy as np 
import pickle

from GPy.models import GPRegression
from GPy.kern import RBF

import warnings
import logging
logging.getLogger('GP').propagate = False

from .model import Model

from ..training import generate_training_data
from ..transform import BoxTransform, MeanTransform
from ..constraints import MeanStateConstraint
from ..approximate_inference import gp_taylor_moment_match

default_noise_var = 1e-5

class GPModel (Model):
	def __init__ (self, candidate_model, moment_match=gp_taylor_moment_match):
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
		super().__init__(candidate_model)

		self.gps = []

		assert candidate_model.u_bounds is not None
		self.u_bounds = candidate_model.u_bounds

		assert candidate_model.x_bounds is not None
		self.x_bounds = candidate_model.x_bounds
		self.x_constraint = None

		if candidate_model.transform is None:
			self.transform = False
		else:
			self.transform = candidate_model.transform
		if self.transform:
			self.z_transform = None
			self.t_transform = None

		if candidate_model.delta_transition is None:
			self.delta_transition = False
		else:
			self.delta_transition = candidate_model.delta_transition

		self.moment_match = moment_match

	"""
	Train GP surrogates
	"""
	def _gp_regression (self, X, Y, kern, **kwargs):
		return GPRegression(X, Y, kern)

	def _train_gp (self, inp, out, active_dims, noise_var=default_noise_var,
	               hyp=None, **kwargs):
		dim  = len( active_dims )
		kern = RBF(dim, active_dims=active_dims, ARD=True)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", 
					message="Your kernel has a different input dimension")
			out = out.reshape((inp.shape[0], 1))
			gp  = self._gp_regression(inp, out, kern, **kwargs)
		if hyp is None:
			# Constrain noise variance
			gp.Gaussian_noise.variance.constrain_fixed(noise_var)
			# Constrain lengthscales
			LS = np.max(inp, axis=0) - np.min(inp, axis=0)
			for d, ad in enumerate(active_dims):
				gp.kern.lengthscale[[d]].constrain_bounded(
					lower=1e-25, upper=10.*LS[ad], warning=False )
			# Optimise hyperparameters
			gp.optimize()
		else:
			# Assign existing hyperparameter values
			gp.update_model(False)
			gp.initialize_parameter()
			gp[:] = hyp
			gp.update_model(True)
		return gp

	def train (self, active_dims, noise_var=default_noise_var, hyp=None, **kwargs):
		dic = {'active_dims': active_dims}
		nom = 'num_data_points_per_num_dim_combo'
		if nom in kwargs:
			dic[nom] = kwargs.get(nom)
		X,U,Z = generate_training_data(self.f,self.x_bounds,self.u_bounds,**dic)
		
		if self.delta_transition:
			Z = [ z - x for z,x in zip(Z,X) ]
		T = [ np.c_[ x, u ] for x,u in zip(X, U) ]

		if self.transform:
			# TODO
			pass

		self._train(T, Z, active_dims, noise_var, hyp=hyp, **kwargs)

	def _train (self, T, Z, active_dims, noise_var=default_noise_var,
	            hyp=None, **kwargs):
		if hyp is None:
			hyp = [None] * len(T)
		for t, z, ad, h in zip(T, Z, active_dims, hyp):
			gp = self._train_gp(t, z, ad, noise_var, hyp=h, **kwargs)
			self.gps.append(gp)


	"""
	Save and load model
	"""
	def save (self, filename): # pragma: no cover
		assert isinstance(filename, str)
		# Filename ending
		suffix = '.doepy'
		lensuf = len(suffix)
		if len(filename) <= lensuf or not filename[-lensuf:] == suffix:
			filename += suffix

		T, Z, active_dims, hyp = [], [], [], []
		for gp in self.gps:
			T.append(gp.X)
			Z.append(gp.Y)
			hyp.append(gp[:])
			active_dims.append(gp.kern.active_dims)

		save_dict = {'T':T, 'Z':Z, 'active_dims':active_dims, 'hyp':hyp}
		with open(filename,'wb') as f:
			pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

	def load (self, filename, **kwargs): # pragma: no cover
		assert isinstance(filename, str)
		# Filename ending
		suffix = '.doepy'
		lensuf = len(suffix)
		if len(filename) <= lensuf or not filename[-lensuf:] == suffix:
			filename += suffix

		assert isfile(filename)
		with open(filename,'rb') as f:
			load_dict = pickle.load(f)

		T = load_dict['T']
		Z = load_dict['Z']
		hyp = load_dict['hyp']
		active_dims = load_dict['active_dims']

		self._train(T, Z, active_dims, hyp=hyp, **kwargs)


	"""
	Transform training data
	"""
	"""
	def _training_data (self, T, Z):
		if not self.transform:
			return T, Z
		self.z_transform = MeanTransform( Z )
		self.t_transform = BoxTransform( T )
		return self.t_transform(T), self.z_transform(Z)
	"""

	"""
	State constraints
	"""
	def initialise_x_constraints (self):
		self.x_constraints = MeanStateConstraint(self.x_bounds)
		self.c, self.dcdU  = None, None

	def update_x_constraints (self, x, s, dxdU, dsdU):
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
		if self.transform:
			assert self.z_transform is not None
			assert self.t_transform is not None
		assert not self.gps == [], 'GP surrogate(s) not trained yet.'

		# Input mean and variance
		tnew = np.array( xk.tolist() + u.tolist() )
		dim  = len( tnew )
		Snew = np.zeros((dim, dim))
		Snew[:self.num_states, :self.num_states] = Sk
		Snew[self.num_states:, self.num_states:] = self.S_u
		if self.transform:
			tnew = self.t_transform(tnew)
			Snew = self.t_transform.cov(Snew)

		# Moment matching
		res = self.moment_match(self.gps, tnew, Snew, grad=grad)
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
