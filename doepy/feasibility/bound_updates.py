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


def stretch_bounds (bounds, alpha=0.2):
	a = alpha / 2.
	A = np.array([ [1+a, -a], [-a, 1+a] ])
	return np.matmul(bounds, A)


class BoundUpdate:
	def __init__ (self, known_dimensions=[], remove_data=True):
		"""
		known_dimensions - list with indices for dimensions with known bounds
		                   (these bounds should not be modified!)
		"""
		self.known_dim = known_dimensions
		self.rm_data   = remove_data

	def __call__ (self, bounds, **kwargs):
		# Update bounds
		bounds, X, y = self._update_unknown_dims(bounds, **kwargs)

		# Remove data outside box
		if self.rm_data:
			I = [i for i,x in enumerate(X) \
				if  np.all( x >= bounds[:,0] ) and np.all( x <= bounds[:,1] )]
			X, y = X[I], y[I]
		return bounds, X, y

	def _update_unknown_dims (self, bounds, **kwargs):
		new_bounds, X, y = self._update(bounds.copy(), **kwargs)
		for d in self.known_dim:
			new_bounds[d] = bounds[d]
		return new_bounds, X, y


class EmpiricalBoxBoundUpdate (BoundUpdate):
	"""
	Return smallest box bounding all observed feasible points
	"""
	def __init__ (self, *args, stretch=0.5, **kwargs):
		super().__init__(*args, **kwargs)
		assert isinstance(stretch, (int,float)) and stretch >= 0
		self.stretch = stretch

	def _update (self, bounds, **kwargs):
		if ('X' in kwargs and 'Y' in kwargs):
			X, y  = kwargs.get('X'), kwargs.get('Y')
		elif not 'model' in kwargs:
			raise ValueError('Need data to update empirical box bounds')
		else:
			model = kwargs.get('model')
			X, y  = model.X.copy(), model.Y[:,0].copy()
		if y.ndim > 1:
			y = y.flatten()
		assert X.shape[0] == len(y), 'Data not of equal length!'

		# Find feasible observations
		I = np.arange(len(y))[y>=0]
		if len(I) > 2*len(bounds):
			# Stretched empirical box
			Xf = X[I]
			for d in range( bounds.shape[0] ):
				if d in self.known_dim: continue
				bounds[d] = np.array([ np.min(Xf[:,d]), np.max(Xf[:,d]) ])
				bounds[d] = stretch_bounds(bounds[d], self.stretch)
		return bounds, X, y
