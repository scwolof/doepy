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

from ..utils import assert_not_none
from ..training import generate_training_data
from ..transform import BoxTransform, Transform
from ..constraints import ConstantMeanStateConstraint

default_noise_var = 1e-5

class GPModel:
	def __init__ (self, transform=None):
		"""
		We assume we do not have gradient information for f
		"""
		self.gps = []

		self.transform = True if transform is None else transform
		self.output_transform = None
		self.input_transform  = None

	"""
	Train GP surrogates
	"""
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
		self.input_transform = BoxTransform( np.concatenate( T, axis=0 ) )
		return [ self.input_transform(t) for t in T ]

	def set_up_target_transforms (self, Z):
		# Target transform: z' ~ N( 0, I )
		z_mean = np.array([ np.mean(z) for z in Z ])
		z_std  = np.array([ np.std(z) for z in Z ])
		self.output_transform = Transform( z_mean, z_std )
		return [ self.output_transform(z,dim=e) for e,z in enumerate(Z) ]

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
				x = self.input_transform(x, back=True)
				z = self.output_transform(z, back=True, dim=e)
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