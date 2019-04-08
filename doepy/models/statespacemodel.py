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

from .model import Model
from ..utils import assert_symmetric_matrix, assert_is_shape

class StateSpaceModel (Model):
	def __init__ (self, candidate_model):
		if candidate_model.num_meas is None:
			candidate_model.num_meas = candidate_model.H.shape[0]
		super().__init__(candidate_model)

		self.H = candidate_model.H
		self.num_states = self.H.shape[1]

		# Process noise
		self.x_covar = candidate_model.x_covar
		if self.x_covar is None:
			self.x_covar = np.zeros(( self.num_states, self.num_states ))

		# Initial states
		self.x0       = candidate_model.x0
		self.x0_covar = candidate_model.x0_covar
		if self.x0_covar is None:
			self.x0_covar = np.zeros(( self.num_states, self.num_states ))

	"""
	Measurement matrix
	"""
	@property
	def H (self):
		return self._H 
	@H.setter
	def H (self, H):
		assert isinstance(H, np.ndarray)
		assert H.ndim == 2
		self._H = H.copy()

	"""
	Process noise covariance matrix
	"""
	@property
	def x_covar (self):
		return self._x_covar
	@x_covar.setter
	def x_covar (self, x_covar):
		assert_symmetric_matrix(x_covar)
		self._x_covar = x_covar.copy()

	"""
	Initial state mean
	"""
	@property
	def x0 (self):
		return self._x0 
	@x0.setter
	def x0 (self, x0):
		assert_is_shape(x0, (self.num_states,))
		self._x0 = x0.copy()

	"""
	Initial state covariance
	"""
	@property
	def x0_covar (self):
		return self._x0_covar 
	@x0_covar.setter
	def x0_covar (self, x0_covar):
		if x0_covar is None:
			x0_covar = np.zeros(( self.num_states, self.num_states ))
		assert_symmetric_matrix(x0_covar)
		self._x0_covar = x0_covar.copy()

	"""
	Latent state constraints
	- For surrogate models that want to enforce state constraints in order
	  not to deviate too far from training data.
	"""
	def initialise_x_constraints (self):
		pass

	def update_x_constraints (self, x, s, dxdU, dsdU, step=None):
		pass

	def get_x_constraints (self):
		return None

	def num_x_constraints (self):
		return 0

	"""
	Function calls
	"""
	def predict_x_dist (self, *args, **kwargs):
		"""
		Return mean and covariance of p( y | f, u, u_covar, p, p_covar )
		"""
		raise NotImplementedError