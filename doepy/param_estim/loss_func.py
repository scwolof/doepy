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
import warnings

from ..utils import assert_is_shape, assert_equal_shape

class Residual:
	r"""
	Weighted square loss:
		L = sum[ (z - y)^T  W  (z - y) ]
	"""
	def __init__ (self, Y, W, invert_W=False):

		if isinstance(Y, (list,tuple)):
			Y  = np.vstack( Y )
		self.Y = Y
		n_meas = self.Y.shape[1]

		if W.ndim == 1:
			if invert_W:
				W = 1./np.where(a>0, a, np.nan)
				W = np.where(np.isnan(W), 0, W )
			self.L = np.diag( np.sqrt(W) )
		else:
			if invert_W:
				try:
					W = np.linalg.inv( W )
				except:
					W = np.eye( n_meas )
					warnings.warn('Could not invert weight matrix. Setting W = I.')
			self.L = np.linalg.cholesky( W ).T
		assert_is_shape(self.L, (n_meas,n_meas))
		
		
	def __call__ (self, Z):
		"""
		Computes  L * (z - y)
		where L^T*L  is the Cholesky decomposition of the weight matrix y_covar
		(which is typically the measurement noise covariance)
		"""
		if isinstance(Z, (list,tuple)):
			Z = np.vstack(Z)
		assert_equal_shape(Z, self.Y)
		return np.matmul(Z - self.Y, self.L)

	def sqrt_lsq_loss (self, Z):
		return self(Z).flatten()

	def sum_lsq_loss (self, Z):
		return np.sum( self(Z)**2 )

