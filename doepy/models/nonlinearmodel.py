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

class NonLinearModel (Model):
	def __init__ (self, f, H, Q, R, x0, num_inputs, P0=None, Su=None, \
					hessian=False):
		"""
		f  : transition function x_{k+1} = f(x_k, u_k)
		H  : observation matrix
		Q  : process noise covariance matrix
		R  : measurement noise covariance
		Su : control input covariance

		Model:
			x_{k+1} = f( x_k, u_k )  +  w_k,   w_k ~ N(0, Q)
				y_k = H * x_k  +  v_k,         v_k ~ N(0, R)
		with 
			x_0 ~ N(x0, P0), u_k ~ N(u_k, Su)

		f is differentiable:
			g, dgdx, dgdu = f( x_k, u_k, grad=True )
			x     [ E ]
			u     [ D ]
			g     [ E ]
			dgdx  [ E x E ]
			dgdu  [ E x D ]

			if hessian: (STRONGLY RECOMMENDED)
			g, dgdx, dgdu, ddgddx, ddgddu = f( x_k, u_k, grad=True )
			ddgddx  [ E x E x E ]
			ddgddu  [ E x D x D ]

			WARNING: STABILITY NOT TESTED WITHOUT HESSIAN INFORMATION
		"""
		super().__init__(f, H, Q, R, x0, num_inputs, P0=P0, Su=Su)
		self.hessian = hessian

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		if self.hessian:
			g, dgdx, dgdu, ddgddx, ddgddu = self.f( xk, u, grad=True )
		else:
			g, dgdx, dgdu = self.f( xk, u, grad=True )
		M  = g
		V  = np.matmul(Sk, dgdx.T)
		St = np.matmul(self.Su, dgdu.T)
		S  = np.matmul(dgdx, V) + np.matmul(dgdu, St) + self.Q
		if not grad:
			return (M, S, V) if cross_cov else (M, S)
		# Compute gradients
		dMdx = dgdx
		dMds = np.zeros([self.num_states]*3)
		dMdu = dgdu
		dSdx = np.zeros([self.num_states]*3)
		if self.hessian:
			dSdx = np.einsum('ik,nkl->inl',V.T,ddgddx) \
					+ np.einsum('nkl,li->nik',ddgddx,V)
		dSds = np.zeros([self.num_states]*4)
		for d1 in range( self.num_states ):
			for d2 in range( self.num_states ):
				dSds[d1,d2] = dgdx[d1][:,None] * dgdx[d2][None,:]
		dSdu = np.zeros(( self.num_states, self.num_states, self.num_inputs ))
		if self.hessian:
			dSdu = np.einsum('ik,nkl->inl',St.T,ddgddu) \
					+ np.einsum('nkl,li->nik',ddgddu,St)
		if not cross_cov:
			return M, S, dMdx, dMds, dMdu, dSdx, dSds, dSdu
		return M, S, V, dMdx, dMds, dMdu, dSdx, dSds, dSdu
