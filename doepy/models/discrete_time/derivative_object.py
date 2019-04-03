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
	def __init__ (self):
		pass

	def insert (self, do, n):
		D = self.__dict__
		for d in D:
			if d in do.__dict__:
				t    = getattr(self,d)
				t[n] = getattr(do,d)

class LatentStateDerivativeObject (DerivativeObject):
	def __init__ (self, model, n=None):
		super().__init__()
		# Dimensions
		D = model.num_inputs
		E = model.num_states
		P = model.num_param
		# Inputs
		x = np.empty( E )        # Latent state mean
		s = np.empty((E, E))     # Latent state covariance
		u = np.empty( D )        # Control mean
		
		# Outputs
		i = (E,) if n is None else (n,E)
		M = np.empty( i )        # Predictive mean
		S = np.empty( i + (E,))  # Predictive covariance
		V = np.empty( i + (E,))  # Predictive state input-output covariance

		self.dMdx = np.zeros( M.shape + x.shape )
		self.dMdu = np.zeros( M.shape + u.shape )
		self.dMds = np.zeros( M.shape + s.shape )
		self.dSdx = np.zeros( S.shape + x.shape )
		self.dSdu = np.zeros( S.shape + u.shape )
		self.dSds = np.zeros( S.shape + s.shape )
		self.dVdx = np.zeros( V.shape + x.shape )
		self.dVdu = np.zeros( V.shape + u.shape )
		self.dVds = np.zeros( V.shape + s.shape )

		P = model.num_param
		if P is not None and P > 0:
			p = np.empty( P )     # Model parameter mean
			self.dMdp = np.zeros( M.shape + p.shape )
			self.dSdp = np.zeros( S.shape + p.shape )
			self.dVdp = np.zeros( V.shape + p.shape )

class MeasDerivativeObject (DerivativeObject):
	def __init__ (self, model, n=None):
		super().__init__()
		# Dimensions
		D = model.num_states
		E = model.num_meas
		# Inputs
		x = np.empty( D )        # Latent state mean
		s = np.empty((D, D))     # Latent state covariance
		
		# Outputs
		i = (E,) if n is None else (n,E)
		M = np.empty( i )        # Predictive mean
		S = np.empty( i + (E,))  # Predictive covariance

		self.dMdx = np.zeros( M.shape + x.shape )
		self.dMds = np.zeros( M.shape + s.shape )
		self.dSdx = np.zeros( S.shape + x.shape )
		self.dSds = np.zeros( S.shape + s.shape )