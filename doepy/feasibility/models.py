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
from scipy.interpolate import griddata

from GPy.models import GPRegression
from GPy.kern import Matern52


class FeasibilityModel:
	def __init__ (self, X, Y, *args, **kwargs):
		self.X = X 
		self.Y = Y.reshape(( self.X.shape[0], 1 ))

	def __call__ (self, Xnew):
		Ynew = griddata(self.X, self.Y, Xnew)
		Ynew[np.isnan(Ynew)] = 0.
		Ynew = Ynew.reshape(( Xnew.shape[0], 1 ))
		return Ynew, 1e-4 * np.ones(Ynew.shape)

	def optimize ():
		pass


class GPFeasibilityModel (GPRegression):
	def __init__ (self, X, Y, hyp=None, noise_var_trainable=False, noise_var=1e-4):
		if hyp is None:
			kern = Matern52(X.shape[1], ARD=True)
		else:
			kern = Matern52(X.shape[1], hyp[0], hyp[1:-1], ARD=True)
			noise_var = hyp[-1]
		super().__init__(X, Y.reshape((X.shape[0], 1)), kern, noise_var=noise_var)
		self.initialize_parameter()
		
		# Noise variance
		self.noise_var_trainable = noise_var_trainable
		self.constant_noise_var  = noise_var
		if not self.noise_var_trainable:
			self.Gaussian_noise.variance.constrain_fixed(self.constant_noise_var)

	def __call__ (self, Xnew):
		return self.predict_noiseless( Xnew )