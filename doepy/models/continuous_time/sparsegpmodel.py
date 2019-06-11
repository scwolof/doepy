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

from GPy.models import SparseGPRegression

from .gpmodel import ctGPModel

class ctSparseGPModel (ctGPModel):
	def __init__ (self, *args, **kwargs):
		"""
		We place a _sparse_ GP prior on the function g
		"""
		super().__init__(*args, **kwargs)

		"""
		Number of training points along each dimension for a grid of training
		data points, i.e. 
		- a 1-dimensional input gets 101 training points
		- a 2-dimensional input gets 21**2 training points
		- a 3-dimensional input gets 10**3 training points
		- etc.
		"""
		self.num_data_points_per_num_dim = [101, 21, 15, 11, 6, 4, 3]

	"""
	Train GP surrogates
	"""
	def _gp_regression (self, X, Y, kern, **kwargs):
		num_default  = [20, 40, 100, 200, 400][np.min((4,len(kern.active_dims)))]
		num_inducing = kwargs.get('num_inducing', num_default)
		return SparseGPRegression(X, Y, kern, num_inducing=num_inducing)