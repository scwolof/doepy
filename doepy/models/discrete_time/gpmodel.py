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

from . import LatentStateDerivativeObject
from .model import dtModel

from ...utils import assert_not_none
from ...training import generate_training_data
from ...transform import BoxTransform, Transform
from ...constraints import ConstantMeanStateConstraint
from ...approximate_inference import gp_taylor_moment_match

default_noise_var = 1e-5

class dtGPModel (dtModel):
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

		assert_not_none(self.u_bounds,'u_bounds')

		assert_not_none(candidate_model.x_bounds,'candidate:x_bounds')
		self.x_bounds      = candidate_model.x_bounds
		self.x_constraints = None

		self.transform   = True if candidate_model.transform is None\
                                else candidate_model.transform
		self.z_transform = None
		self.t_transform = None

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
		if self.num_param is not None and self.num_param > 0:
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

	def _train (self, T, Z, active_dims, noise_var=default_noise_var,
	            hyp=None, **kwargs):
		if self.transform:
			T = self.set_up_input_transforms(T)
			Z = self.set_up_target_transforms(Z)

		if hyp is None:
			hyp = [None] * len(T)

		self.gps = []
		for t, z, ad, h in zip(T, Z, active_dims, hyp):
			gp = self._train_gp(t, z, ad, noise_var, hyp=h, **kwargs)
			self.gps.append(gp)

	def set_up_input_transforms (self, T):
		# Input transforms: t' ~ [ 0, 1 ]^D
		self.t_transform = BoxTransform( np.concatenate( T, axis=0 ) )
		return [ self.t_transform(t) for t in T ]

	def set_up_target_transforms (self, Z):
		# Target transform: z' ~ N( 0, I )
		z_mean = np.array([ np.mean(z) for z in Z ])
		z_std  = np.array([ np.std(z) for z in Z ])
		self.z_transform = Transform( z_mean, z_std )
		return [ self.z_transform(z,dim=e) for e,z in enumerate(Z) ]

	def _train_gp (self, inp, out, active_dims, noise_var=default_noise_var,
	               hyp=None, **kwargs):
		kern = RBF(len(active_dims), active_dims=active_dims, ARD=True)

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", 
					message="Your kernel has a different input dimension")
			out = out.reshape((inp.shape[0], 1))
			gp  = self._gp_regression(inp, out, kern, **kwargs)

		# Constrain noise variance
		gp.Gaussian_noise.variance.constrain_fixed(noise_var)
		# Constrain lengthscales
		LS = np.max(inp, axis=0) - np.min(inp, axis=0)
		for d, ad in enumerate(active_dims):
			gp.kern.lengthscale[[d]].constrain_bounded(
				lower=1e-25, upper=10.*LS[ad], warning=False )

		if hyp is None:
			# Optimise hyperparameters
			gp.optimize()
		else:
			# Assign existing hyperparameter values
			gp.update_model(False)
			gp.initialize_parameter()
			gp[:] = hyp
			gp.update_model(True)
		return gp

	def _gp_regression (self, X, Y, kern, **kwargs):
		return GPRegression(X, Y, kern)


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
		for e,gp in enumerate(self.gps):
			x, z = gp.X, gp.Y
			if self.transform:
				x = self.t_transform(x, back=True)
				z = self.z_transform(z, back=True, dim=e)
			T.append(x)
			Z.append(z)
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
				hyp[e,a] = self.t_transform(l, back=True, dim=e)
		hyp = np.max(hyp,axis=0)

		# Find min distances in training data
		Xt = np.vstack([ gp.X for gp in self.gps ])
		Xt = self.t_transform(Xt, back=True)[:,:self.num_states]
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
		if self.transform:
			assert_not_none(self.z_transform, 'z_transform')
			assert_not_none(self.t_transform, 'z_transform')
		assert not self.gps == [], 'GP surrogate(s) not trained yet.'

		# Input mean
		tnew = np.array( xk.tolist() + u.tolist() )
		nums = [ self.num_states, self.num_inputs ]  # List of no. of dims
		Ss   = [Sk, self.u_covar]                    # List of covariances
		if self.num_param is not None and self.num_param > 0:
			assert_not_none(self.p_mean, '%s:p_mean'%self.name)
			assert_not_none(self.p_covar, '%s:p_covar'%self.name)
			tnew  = np.concatenate(( tnew, self.p_mean ))
			nums += [ self.num_param ]
			Ss   += [ self.p_covar ]

		# Input variance
		dim = len( tnew )
		S_t = np.zeros(( dim, dim ))
		Dss = np.cumsum( [0] + nums )
		for i,Si in enumerate( Ss ):  # Fill block-diagonal matrix
			i1, i2 = Dss[i], Dss[i+1]
			S_t[i1:i2, i1:i2] = Si

		if self.transform:
			tnew = self.t_transform( tnew )
			S_t  = self.t_transform.cov( S_t )

		# Moment matching
		res = self.moment_match(self.gps, tnew, S_t, grad=grad)
		if grad:
			M, S, V, domm = res
			do = LatentStateDerivativeObject(self)
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
				domm.dMdm *= qz[:,None] / qt[None,:]
				domm.dMds *= qz[:,None,None] / qtqt[None,:,:]
				domm.dSdm *= qzqz[:,:,None] / qt[None,None,:]
				domm.dSds *= qzqz[:,:,None,None] / qtqt[None,None,:,:]
				domm.dVdm *= qtqz[:,:,None] / qt[None,None,:]
				domm.dVds *= qtqz[:,:,None,None] / qtqt[None,None,:,:]

		# Separate state and control dimensions again
		V = V[:self.num_states]
		if grad:
			D  = self.num_states
			dn = self.num_states + self.num_inputs
			do.dMdx = domm.dMdm[:,:D]
			do.dMdu = domm.dMdm[:,D:dn]
			do.dMds = domm.dMds[:,:D,:D]
			do.dSdx = domm.dSdm[:,:,:D]
			do.dSdu = domm.dSdm[:,:,D:dn]
			do.dSds = domm.dSds[:,:,:D,:D]
			do.dVdx = domm.dVdm[:D,:,:D]
			do.dVdu = domm.dVdm[:D,:,D:dn]
			do.dVds = domm.dVds[:D,:,:D,:D]
			if self.num_param is not None and self.num_param > 0:
				P = self.num_param
				do.dMdp = domm.dMdm[:,-P:]
				do.dSdp = domm.dSdm[:,:,-P:]
				do.dVdp = domm.dVdm[:,:,-P:]

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
		if grad:
			ret += (do,)
		return ret
