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

from .model import dtModel

class dtLinearModel (dtModel):
	def __init__ (self, candidate_model):
		"""
		Model:
			f(x_k, u_k) = F * x_k  +  B * u_k
		"""
		assert candidate_model.F is not None
		assert candidate_model.B is not None
		F, B = candidate_model.F, candidate_model.B

		candidate_model.f = lambda x,u: np.matmul(F,x) + np.matmul(B,u)
		super().__init__(candidate_model)

		self.F = F
		self.B = B

	"""
	Transition matrix
	"""
	@property
	def F (self):
		return self._F
	@F.setter
	def F (self, F):
		assert isinstance(F, np.ndarray)
		assert F.shape == (self.num_states, self.num_states)
		self._F  = F.copy()
	@F.deleter
	def F (self):
		del self._F

	"""
	Control input matrix
	"""
	@property
	def B (self):
		return self._B
	@B.setter
	def B (self, B):
		assert isinstance(B, np.ndarray)
		assert B.shape[0] == self.num_states
		self._B  = B.copy()
	@B.deleter
	def B (self):
		del self._B

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		M = np.matmul(self.F, xk) + np.matmul(self.B, u)
		V = np.matmul(Sk, self.F.T)
		S = np.matmul(self.u_covar, self.B.T)
		S = np.matmul(self.F, V) + np.matmul(self.B, S) + self.x_covar
		if not grad:
			return (M, S, V) if cross_cov else (M, S)
		# Compute gradients
		dMdx = self.F.copy()
		dMds = np.zeros([self.num_states]*3)
		dMdu = self.B.copy()
		dSdx = np.zeros([self.num_states]*3)
		dSds = np.zeros([self.num_states]*4)
		for d1 in range( self.num_states ):
			for d2 in range( self.num_states ):
				dSds[d1,d2] = self.F[d1][:,None] * self.F[d2][None,:]
		dSdu = np.zeros(( self.num_states, self.num_states, self.num_inputs ))
		if not cross_cov:
			return M, S, dMdx, dMds, dMdu, dSdx, dSds, dSdu

		# Compute cross-covariance
		#dVdx = np.zeros([self.num_states]*3)
		#dVds = np.zeros([self.num_states]*4)
		#for d1 in range( self.num_states ):
		#	dVds[d1,:,d1] = self.F.copy()
		#dVdu = np.zeros(( self.num_states, self.num_states, self.num_inputs ))

		return M, S, V, dMdx, dMds, dMdu, dSdx, dSds, dSdu
