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

from ..utils import assert_symmetric_matrix, assert_is_shape

class Model:
	def __init__ (self, candidate_model):
		"""
		Model:
			y = f( u, p )  +  v,   
		with 
			u ~ N(u, u_covar), 
			p ~ N(p, p_covar), 
			v ~ N(0, y_covar)
		"""
		self.name = candidate_model.name
		if self.name is None:
			self.name = 'Model' + np.random.randint(1000)

		# Model function
		self.f = candidate_model.f

		# Control signal u
		self.num_inputs = candidate_model.num_inputs
		assert isinstance(self.num_inputs, int) and self.num_inputs >  0
		self.u_bounds   = candidate_model.u_bounds
		self.u_covar    = candidate_model.u_covar
		if self.u_covar is None:
			self.u_covar = np.zeros(( self.num_inputs, self.num_inputs ))

		# Model parameters p
		self.num_param = candidate_model.num_param
		if not self.num_param is None:
			assert isinstance(self.num_param, int) and self.num_param >= 0
			#self.p_mean  = candidate_model.p_mean
			#self.p_covar = candidate_model.p_covar

		# Observations y 
		self.num_meas = candidate_model.num_meas
		assert isinstance(self.num_meas, int) and self.num_meas > 0
		self.y_covar  = candidate_model.y_covar
		if self.y_covar is None:
			self.y_covar = np.zeros(( self.num_meas, self.num_meas ))



	"""
	Model function
	"""
	@property
	def f (self):
		return self._f 
	@f.setter
	def f (self, f):
		assert callable(f)
		self._f = f

	"""
	Measurement noise covariance matrix
	"""
	@property
	def y_covar (self):
		return self._y_covar
	@y_covar.setter
	def y_covar (self, y_covar):
		assert_is_shape(y_covar, (self.num_meas, self.num_meas))
		assert_symmetric_matrix(y_covar)
		self._y_covar = y_covar.copy()

	"""
	Input covariance
	"""
	@property
	def u_covar (self):
		return self._u_covar
	@u_covar.setter
	def u_covar (self, u_covar):
		if u_covar is None:
			u_covar = np.zeros(( self.num_inputs, self.num_inputs ))
		assert_is_shape(u_covar, (self.num_inputs, self.num_inputs))
		assert_symmetric_matrix(u_covar)
		self._u_covar = u_covar.copy()

	"""
	Model parameter mean
	"""
	@property
	def p_mean (self):
		return self._p_mean
	@p_mean.setter
	def p_mean (self, p_mean):
		assert_is_shape(p_mean, (self.num_param,))
		self._p_mean = p_mean.copy()

	"""
	Model parameter covariance
	"""
	@property
	def p_covar (self):
		return self._p_covar
	@p_covar.setter
	def p_covar (self, p_covar):
		if p_covar is None:
			p_covar = np.zeros(( self.num_param, self.num_param ))
		assert_is_shape(p_covar, (self.num_param, self.num_param))
		assert_symmetric_matrix(p_covar)
		self._p_covar = p_covar.copy()

	"""
	Function calls
	"""
	def predict (self, *args, **kwargs):
		return self.f(*args, **kwargs)

	def predict_y_dist (self, *args, **kwargs):
		"""
		Return mean and covariance of p( y | f, u, u_covar, p, p_covar )
		"""
		raise NotImplementedError