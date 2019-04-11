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
from numpy.random import multivariate_normal as mvn

from ..statespacemodel import StateSpaceModel

class dtModel (StateSpaceModel):
	def __init__ (self, candidate_model):
		"""
		Model:
			x_{k+1} = f( x_k, u_k, p )  +  w_k,   w_k ~ N(0, x_covar)
			    z_k = H * x_k
			    y_k = z_k  +  v_k,                v_k ~ N(0, y_covar)
		with 
			x_0 ~ N(x0, x0_covar), 
			u_k ~ N(u_k, u_covar), 
			p   ~ N(p, p_covar)
			u_k of dimension num_inputs
		"""
		super().__init__(candidate_model)

	"""
	Function calls
	"""
	def _predict (self, x, u):
		args = (x, u) if self.num_param < 1 else (x, u, self.p_mean)
		xk1  = self.f(*args)
		yk   = np.matmul(self.H, x)
		return xk1, yk

	def _sample (self, x, u):
		us = mvn(u, self.u_covar)
		xy = self.predict(x, us)
		wk = mvn( np.zeros(self.num_states), self.x_covar )
		vk = mvn( np.zeros(self.num_meas), self.y_covar )
		return xy[0] + wk, xy[1] + vk