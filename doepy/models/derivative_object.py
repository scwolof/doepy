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

class DerivativeObject:
	def __init__ (self, num_out, num_inputs=None, num_param=None, num_states=None, 
		Hessian=False, num_test_points=None, covariance=False, 
		in_out_covariance=False):

		has_inputs = num_inputs is not None and num_inputs > 0
		has_param  = num_param is not None and num_param > 0
		has_states = num_states is not None and num_states > 0

		# Outputs
		i = (num_out,) if num_test_points is None else (num_test_points, num_out)
		M = np.empty( i )                  # Predictive mean
		if covariance:
			S = np.empty( i + (num_out,))  # Predictive covariance
		if in_out_covariance:
			V = np.empty( i + (num_out,))  # Predictive input-output covariance

		# Control input
		if has_inputs:
			u = np.empty( num_inputs )
			self.dMdu = np.zeros( M.shape + u.shape )
			if covariance:
				self.dSdu = np.zeros( S.shape + u.shape )
			if in_out_covariance:
				self.dVdu = np.zeros( V.shape + u.shape )
			if Hessian:
				self.dMduu = np.zeros( M.shape + u.shape + u.shape )

		if has_param:
			p = np.empty( num_param )              # Model parameter mean
			self.dMdp = np.zeros( M.shape + p.shape )
			if covariance:
				self.dSdp = np.zeros( S.shape + p.shape )
			if in_out_covariance:
				self.dVdp = np.zeros( V.shape + p.shape )
			if Hessian:
				self.dMdpp = np.zeros( M.shape + p.shape + p.shape )
				if has_inputs:
					self.dMdup = np.zeros( M.shape + u.shape + p.shape )

		if has_states:
			x = np.empty( num_states )             # Latent state mean
			s = np.empty((num_states, num_states)) # Latent state covariance

			self.dMdx = np.zeros( M.shape + x.shape )
			self.dMds = np.zeros( M.shape + s.shape )
			if covariance:
				self.dSdx = np.zeros( S.shape + x.shape )
				self.dSds = np.zeros( S.shape + s.shape )
			if in_out_covariance:
				self.dVdx = np.zeros( V.shape + x.shape )
				self.dVds = np.zeros( V.shape + s.shape )

			if Hessian:
				self.dMdxx = np.zeros( M.shape + x.shape + x.shape )
				if has_inputs:
					self.dMdxu = np.zeros( M.shape + x.shape + u.shape )
				if has_param:
					self.dMdxp = np.zeros( M.shape + x.shape + p.shape )

	def insert (self, do, n):
		D = self.__dict__
		for d in D:
			if d in do.__dict__:
				t    = getattr(self,d)
				t[n] = getattr(do,d)


class LatentStateDerivativeObject (DerivativeObject):
	def __init__ (self, model, num_test_points=None, Hessian=False):
		d = {'num_out': model.num_states,
		     'num_inputs': model.num_inputs,
		     'num_param': model.num_param,
		     'num_states': model.num_states,
		     'covariance': True,
		     'in_out_covariance':True,
		     'num_test_points':num_test_points}
		super().__init__(**d)


class MeasDerivativeObject (DerivativeObject):
	def __init__ (self, model, num_test_points=None):
		d = {'num_out': model.num_meas,
		     'num_states': model.num_states,
		     'covariance': True,
		     'num_test_points':num_test_points}
		super().__init__(**d)