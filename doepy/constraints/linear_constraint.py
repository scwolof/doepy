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
from scipy.linalg import block_diag

from ..utils import assert_is_instance, assert_is_shape


class LinearConstraint:
	"""
	Class for linear constraints of the type
		A * m - b >= 0

	Attributes:
		A 	[ n, d ] numpy.ndarray
		b 	[ n, ] numpy.ndarray
		n 	(int) number of constraints
		d 	(int) dimensionality
	"""

	def __init__ (self, A, b):
		"""
		Initialise linear constraints object

		Inputs:
			A 	[ n, d ] numpy.ndarray
			b 	[ n, ] numpy.ndarray
		"""
		self.A = A
		self.b = b

		assert_is_instance(self.A, np.ndarray, 'A')
		assert_is_instance(self.b, np.ndarray, 'b')

		if self.A.ndim == 1:
			self.A = self.A.reshape((1,-1))
		assert self.A.ndim == 2, 'A must be 2D numpy array. Given shape %s'%A.shape
		self.n = self.A.shape[0]
		self.d = self.A.shape[1]

		if self.b.ndim > 1:
			self.b = self.b.flatten()
		assert_is_shape(b, (self.n,))


	def __call__ (self, m, grad=False):
		"""
		Evaluate linear constraint
			A * m - b >= 0

		Computes A * m - b. Negative elements in the result indicate the
		constraint is violated.

		Inputs:
			m 		[ (N), d ] numpy.ndarray, evaluation point
			grad 	(boolean, optional) return gradients

		Returns:
			c 		[ (N), n ] numpy.ndarray, constraint c = A * m - b
			dc 		(if grad) [ (N), n, d ] derivatives d c / d m
		"""
		assert_is_instance(m, np.ndarray, 'input')
		if m.ndim > 2:
			raise ValueError('input must be 1D or 2D array, got %s'%m.shape)

		if m.ndim == 1:
			# 1D input m
			assert_is_shape(m, (self.d,))

			c  = np.matmul(self.A, m) - self.b
			if grad:
				dc = self.A
		else:
			# 2D input m
			assert_is_shape(m, (-1, self.d))
			c  = np.matmul(m, self.A.T) - self.b
			if grad:
				dc = np.array([self.A for _ in m])
		
		return c if not grad else (c, dc)


	def num_constraint (self):
		"""
		Returns:
			(int) number of constraints
		"""
		return self.n


	def __add__ (self, l2):
		"""
		Add linear constraints together to create new LinearConstraint object
		of dimensionality [n1+...+nn, d]

			A1 * x - b1 >= 0 	(n1 constraints)
			...
		+   An * x - bn >= 0	(nn constraints)
		--------------------------------------------
			|  A1 |       |  b1 |
		=	| ... | * x - | ... |  >= 0    (n1 + ... + nn constraints)
			|  An |       |  bn |

		Inputs:
			l2 		LinearConstraint object

		Returns:
			Combined LinearConstraint object
		"""
		if isinstance(l2, int):
			return self
		assert_is_instance(l2, LinearConstraint, 'l2')
		assert self.d == l2.d, 'Dimensions must match: %d != %d' %(self.d, l2.d)

		A = np.vstack((self.A, l2.A))
		b = np.concatenate((self.b, l2.b))
		return LinearConstraint(A, b)


	def __radd__ (self, l2):
		return self.__add__(l2)


	def __mul__ (self, N):
		"""
		Repeat the constraint N times, to create a new LinearConstraint object
		of dimensionality [N*n, N*d]

			A * x - b1 >= 0 	(n constraints)
		*   N
		--------------------------------------------
			|  A   0  ... |       |  b  |
		=	|  0   A  ... | * x - | ... |  >= 0    ( n * N constraints)
			| ... ...  A  |       |  b  |
		
		"""
		assert_is_instance(N, int, 'N')

		A = block_diag(*([ self.A ] * N))
		b = np.concatenate([ self.b ] * N)
		return LinearConstraint(A, b)