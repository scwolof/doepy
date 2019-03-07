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


def stretch_bounds (bounds, alpha=0.2):
	a = alpha / 2.
	A = np.array([ [1+a, -a], [-a, 1+a] ])
	return np.matmul(bounds, A)


class BoundUpdate:
	def __init__ (self, known_dimensions):
		"""
		known_dimensions - list with indices for dimensions with known bounds
		                   (these bounds should not be modified!)
		"""
		self.dim = known_dimensions

	def __call__ (self, bounds, X, Y, i=None, remove_data=True):
		# Update bounds
		bounds = self._update_unknown_dims(bounds, X, Y, i=i)

		# Remove data outside box
		if not remove_data:
			return bounds, X, Y
		I = [i for i,x in enumerate(X) \
		        if  np.all( x >= bounds[:,0] ) and np.all( x <= bounds[:,1] )]
		return bounds, X[I], Y[I]

	def _update_unknown_dims (self, bounds, X, Y, i=None):
		new_bounds = self._update(bounds.copy(), X, Y, i=i)
		for d in self.dim:
			new_bounds[d] = bounds[d]
		return new_bounds


class EmpiricalBoxBoundUpdate (BoundUpdate):
	def __init__ (self, known_dimensions, stretch=0.5):
		super().__init__(known_dimensions)
		assert isinstance(stretch, (int,float)) and stretch >= 0
		self.stretch = stretch

	def _update (self, bounds, X, Y, i=None):
		# Find feasible observations
		I = np.arange(len(Y))[Y>=0]
		if len(I) <= 2*len(bounds):
			return bounds
		Xf = X[I]
		# Stretched empirical box
		for d in range( bounds.shape[0] ):
			if d in self.dim: continue
			bounds[d] = np.array([ np.min(Xf[:,d]), np.max(Xf[:,d]) ])
			bounds[d] = stretch_bounds(bounds[d], self.stretch)
		return bounds
