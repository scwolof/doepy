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

class CandidateModel:
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

		# Transition function handle
		self.f = assign_variable('f')
		# State transition matrix for linear models
		self.F = assign_variable('F', np.ndarray)
		# Control transition matrix for linear models
		self.B = assign_variable('B', np.ndarray)
		# Observation matrix
		self.H = assign_variable('H', np.ndarray)
		# Process noise covariance matrix
		self.Q = assign_variable('Q', np.ndarray)
		# Measurement noise covariance matrix
		self.R = assign_variable('R', np.ndarray)
		# Model parameter mean
		self.p = assign_variable('p', np.ndarray)
		# Initial state mean
		self.x0 = assign_variable('x0', np.ndarray)
		# Control input covariance
		self.S_u = assign_variable('S_u', np.ndarray)
		# Model parameter covariance
		self.S_p = assign_variable('S_p', np.ndarray)
		# Initial state covariance
		self.S_x0 = assign_variable('S_x0', np.ndarray)
		# Candidate model name
		self.name = assign_variable('name', str)
		# Transition function handle
		self.hessian = assign_variable('hessian', bool)
		# Bounds on latent states - for data-driven surrogate models
		self.x_bounds = assign_variable('x_bounds', np.ndarray)
		# Normalise inputs and training data in surrogate models
		self.transform = assign_variable('transform', bool)
		# Number of control inputs
		self.num_inputs = assign_variable('num_inputs', int)
		# Delta transition: f(x,u,p) = x + f_delta(x,u,p)
		self.delta_transition = assign_variable('delta_transition', bool)
