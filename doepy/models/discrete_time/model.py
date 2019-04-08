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
from numpy.random import multivariate_normal as mvn

from ..statespacemodel import StateSpaceModel

class dtModel (StateSpaceModel):
	def __init__ (self, candidate_model):
		"""
		Model:
			x_{k+1} = f( x_k, u_k, p )  +  w_k,   w_k ~ N(0, x_covar)
			    z_k = H * x_k
			    y_k = z_k  +  v_k,                v_k ~ N(0, y_covar)
		with 
			x_0 ~ N(x0, x0_covar), 
			u_k ~ N(u_k, u_covar), 
			p   ~ N(p, p_covar)
			u_k of dimension num_inputs
		"""
		super().__init__(candidate_model)

	"""
	Function calls
	"""
	def _predict (self, x, u):
		args = (x, u) if self.num_param < 1 else (x, u, self.p_mean)
		xk1  = self.f(*args)
		yk   = np.matmul(self.H, x)
		return xk1, yk

	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		# Implemented in children classes (e.g. LinearModel)
		raise NotImplementedError

	def filter (self, yk, x, s):
		"""
		Input observation yk, prediction mean m and variance s 
			p( x_k | y_{1 : k-1} ) = N( x, s )
		Outputs state posterior mean xk and variance sk 
			p( x_k | y_{1 : k} ) = N( xk, sk )

		yk : [ (n), E ]
		x  : [ (n), D ]
		s  : [ (n), D, D ]
		"""
		assert yk.ndim == x.ndim

		if yk.ndim == 1:
			return self._filter(yk, x, s)

		n = len(yk)
		X = np.zeros(( n, self.num_states ))
		S = np.zeros(( n, self.num_states, self.num_states ))
		for k in range(n):
			X[k], S[k] = self._filter(yk[k], x[k], s[k])
		return X, S

	def _filter (self, yk, x, s):
		sH = np.matmul(s, self.H.T)
		K  = np.matmul(sH, np.linalg.inv(np.matmul(self.H, sH) + self.y_covar))
		xk = x + np.matmul(K, yk - np.matmul(self.H, x))
		sk = s - np.matmul(K, sH.T)
		return xk, sk

	def predict_filter (self, Y, x0, x0_covar, U):
		"""
		Filter sequence, based on observations Y, controls U,
		and with prediction p(x_1) ~ N(x0, x0_covar)

		Y        : [ n, E ]       ( y_1, ..., y_n )
		x0       : [ D, ]
		x0_covar : [ D, D ]
		U        : [ n, D_U ]     ( u_1, ..., u_{n-1} )

		Outputs
		x  : [ n+1, D ]       ( x_1, ..., x_n )
		s  : [ n+1, D, D]
		"""
		n = len(Y)
		x = np.zeros(( n, self.num_states ))
		s = np.zeros(( n, self.num_states, self.num_states ))
		M, S = x0, x0_covar
		for k in range( n ):
			x[k], s[k] = self._filter(Y[k], M, S)
			if k < n-1:
				M, S = self.predict_x_dist(x[k], s[k], U[k])
		return x, s

	def smooth (self, X, S, U):
		"""
		Iteratively smooths sequences X = [x_1, ..., x_n] and 
		S = [S_1, ..., S_n] with controls U = [u_1, ..., u_n]
		"""
		n  = len(X)
		Xs = np.zeros(( n, self.num_states ))
		Ss = np.zeros(( n, self.num_states, self.num_states ))
		Xs[-1] = X[-1]
		Ss[-1] = S[-1]
		for k in np.arange(1, n)[::-1]:
		    Xs[k-1],Ss[k-1] = self._smooth(X[k-1], S[k-1], Xs[k], Ss[k], U[k-1])
		return Xs, Ss

	def _smooth (self, xk, Sk, xk1, Sk1, uk):
		"""
		Inputs xk, Sk, xk1, S_{k+1}, uk
			p( x_k | y_{1:k} ) = N( xk, Sk )
			p( x_{k+1} | y_{1:T}, uk ) = N( xk1, Sk1 )

		Produces smoothed x_k, S_k
			p( x_k | y_{1:T} ) = N( xs, Ss )
		"""
		m, S, V = self.predict_x_dist(xk, Sk, uk, cross_cov=True)
		J  = np.matmul( V, np.linalg.inv(S) )
		xs = xk + np.matmul( J, xk1 - m )
		Ss = Sk + np.matmul( J, np.matmul( Sk1 - S, J.T ) )
		return xs, Ss

