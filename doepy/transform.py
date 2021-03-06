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

class Transform:
	"""
	X  ->  ( X - self.m ) / self.q 
	"""
	def __init__ (self, m, q):
		self.m = m
		self.q = q
		self.qsafe0 = np.where(self.q != 0, 1, 0)
		self.qsafe1 = np.where(self.q != 0, self.q, 1)

	def __call__ (self, X, back=False, dim=None):
		if dim is None:
			if back:
				return self.m + X * self.q
			return (X - self.m) / self.qsafe1 * self.qsafe0
		else:
			if back:
				return self.m[dim] + X * self.q[dim]
			return (X - self.m[dim]) / self.qsafe1[dim] * self.qsafe0[dim]

	def var (self, X, back=False):
		if back:
			return X * self.q**2
		return X / self.qsafe1**2 * self.qsafe0

	def cov (self, X, back=False):
		if back:
			return X * (self.q[:,None] * self.q[None,:])
		Q0 = (self.qsafe0[:,None] * self.qsafe0[None,:])
		Q1 = (self.qsafe1[:,None] * self.qsafe1[None,:])
		return X / Q1 * Q0

	def prediction (self, M, S, back=False):
		Mt = self(M, back=back)
		if S.ndim == 2:
			return Mt, self.var(S, back=back)
		return Mt, self.cov(S, back=back)


class BoxTransform (Transform):
	"""
	X  ->  ( X - xmin ) / ( xmax - xmin ), i.e. X' in [0, 1]
	"""
	def __init__ (self, X):
		xmin = np.min(X, axis=0)
		xmax = np.max(X, axis=0)
		super(BoxTransform, self).__init__(xmin, xmax-xmin)


class MeanTransform (Transform):
	"""
	X  ->  ( X - mean(X) ) / std( X ), i.e. X' in N(0, 1)
	"""
	def __init__ (self, X):
		mean = np.mean(X, axis=0)
		std  = np.std(X, axis=0)
		super(MeanTransform, self).__init__(mean, std)
