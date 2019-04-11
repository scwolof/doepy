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

from ..utils import assert_is_instance

class CandidateWrapper:
	def __init__ (self, **kwargs):

		# Function call to assign variable with given name
		# and check that the data type is correct
		def assign_variable (name, dtype=None, default=None):
			value = kwargs.get(name, None)
			if not (value is None or dtype is None):
				assert_is_instance(value, dtype, name)
			setattr(self, name, value)

		########################
		## Common
		########################
		# Transition function handle
		assign_variable('f')
		# Candidate model name
		assign_variable('name', str)

		# Dimensionality of observations
		assign_variable('num_meas', int)
		# Measurement noise covariance matrix
		assign_variable('y_covar', np.ndarray)

		# Number of control inputs
		assign_variable('num_inputs', int)
		# Control input covariance
		assign_variable('u_covar', np.ndarray)
		# Bounds on control inputs
		assign_variable('u_bounds', np.ndarray)

		# Number of model variables
		assign_variable('num_param', int)
		# Model parameter mean
		assign_variable('p_mean', np.ndarray)
		# Model parameter covariance
		assign_variable('p_covar', np.ndarray)

		# Normalise inputs and training data in surrogate models
		assign_variable('transform', bool)

		# Model function handle return first- and second-order derivatives
		assign_variable('hessian', bool)


		########################
		## State-space models
		########################
		# Observation matrix
		assign_variable('H', np.ndarray)

		# State transition matrix for linear models
		assign_variable('F', np.ndarray)
		# Control transition matrix for linear models
		assign_variable('B', np.ndarray)

		# Initial state mean
		assign_variable('x0', np.ndarray)
		# Initial state covariance
		assign_variable('x0_covar', np.ndarray)

		# Process noise covariance matrix
		assign_variable('x_covar', np.ndarray)
		# Bounds on latent states - for data-driven surrogate models
		assign_variable('x_bounds', np.ndarray)


		######################################
		## Discrete-time state space models
		######################################
		# Delta transition: f(x,u,p) = x + f_delta(x,u,p)
		assign_variable('delta_transition', bool)


		######################################
		## Continuous-time state space models
		######################################
		# Time between measurements
		assign_variable('step_length', float)
		# Num time segments in [0, step_length] for ODE solver
		assign_variable('time_steps', int)
