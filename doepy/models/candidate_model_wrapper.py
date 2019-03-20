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

class CandidateWrapper:
	def __init__ (self, **kwargs):

		# Function call to assign variable with given name
		# and check that the data type is correct
		def assign_variable (name, dtype=None, default=None):
			x = kwargs.get(name, None)
			if x is None:
				return None
			if not dtype is None:
				assert isinstance(x, dtype),\
				    '%s must be of type %s: %s'%(name, dtype, x)
			return x

		########################
		## Common
		########################
		# Transition function handle
		self.f = assign_variable('f')
		# Candidate model name
		self.name = assign_variable('name', str)

		# Dimensionality of observations
		self.num_meas = assign_variable('num_inputs', int)
		# Measurement noise covariance matrix
		self.y_covar  = assign_variable('y_covar', np.ndarray)

		# Number of control inputs
		self.num_inputs = assign_variable('num_inputs', int)
		# Control input covariance
		self.u_covar    = assign_variable('u_covar', np.ndarray)
		# Bounds on control inputs
		self.u_bounds   = assign_variable('u_bounds', np.ndarray)

		# Number of model variables
		self.num_param = assign_variable('num_param', int)
		# Model parameter mean
		self.p_mean    = assign_variable('p_mean', np.ndarray)
		# Model parameter covariance
		self.p_covar   = assign_variable('p_covar', np.ndarray)

		# Normalise inputs and training data in surrogate models
		self.transform = assign_variable('transform', bool)

		# Model function handle return first- and second-order derivatives
		self.hessian = assign_variable('hessian', bool)


		########################
		## Discrete-time models
		########################
		# Observation matrix
		self.H = assign_variable('H', np.ndarray)

		# State transition matrix for linear models
		self.F = assign_variable('F', np.ndarray)
		# Control transition matrix for linear models
		self.B = assign_variable('B', np.ndarray)

		# Initial state mean
		self.x0       = assign_variable('x0', np.ndarray)
		# Initial state covariance
		self.x0_covar = assign_variable('S_x0', np.ndarray)

		# Process noise covariance matrix
		self.x_covar  = assign_variable('x_covar', np.ndarray)
		# Bounds on latent states - for data-driven surrogate models
		self.x_bounds = assign_variable('x_bounds', np.ndarray)

		# Delta transition: f(x,u,p) = x + f_delta(x,u,p)
		self.delta_transition = assign_variable('delta_transition', bool)
