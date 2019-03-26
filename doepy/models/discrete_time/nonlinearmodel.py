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
from ...approximate_inference import taylor_moment_match

class dtNonLinearModel (dtModel):
	def __init__ (self, candidate_model):
		"""
		Transition function f is differentiable:
		    g, dgdx, dgdu = f( x_k, u_k, grad=True )
		    x_k   [ E ]
		    u_k   [ D ]
		    g     [ E ]
		    dgdx  [ E x E ]
		    dgdu  [ E x D ]

		    if hessian: (STRONGLY RECOMMENDED)
		    g, dgdx, dgdu, ddgddx, ddgddu, ddgdxu = f( x_k, u_k, grad=True )
		    ddgddx  [ E x E x E ]
		    ddgddu  [ E x D x D ]
		    ddgdxu  [ E x E x D ]

		    WARNING: NOT PROPERLY TESTED WITHOUT HESSIAN INFORMATION
		"""
		super().__init__(candidate_model)

		if candidate_model.hessian is None:
			self.hessian = False
		else:
			self.hessian = candidate_model.hessian

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		if self.hessian:
			M, dfdx, dfdu, ddfddx, ddfddu, ddfdxu = self.f( xk, u, grad=True )
			x_xu   = np.concatenate(( ddfddx, ddfdxu ), axis=2)
			ddfdxu = np.transpose(ddfdxu, axes=[0,2,1])
			ux_u   = np.concatenate(( ddfdxu, ddfddu ), axis=2)
			ddM    = np.concatenate(( x_xu, ux_u ), axis=1)
		else:
			M, dfdx, dfdu = self.f( xk, u, grad=True )
			ddM = None
		dMdm = np.concatenate((dfdx, dfdu), axis=1)
		dim  = self.num_states + self.num_inputs
		Snew = np.zeros((dim, dim))
		Dss  = np.cumsum([0] + [self.num_states, self.num_inputs]) #, self.num_param])
		for i,Si in enumerate([Sk, self.u_covar]): #, self.p_covar]):
			i1, i2 = Dss[i], Dss[i+1]
			Snew[i1:i2, i1:i2] = Si

		if not grad:
			S, V = taylor_moment_match(Snew, dMdm)
			S   += self.x_covar
			V    = V[:self.num_states]
			return (M, S, V) if cross_cov else (M, S)

		S,V,dMds,dSdm,dSds,dVdm,dVds = taylor_moment_match(Snew, dMdm, ddM, True)

		S   += self.x_covar
		V    = V[:self.num_states]
		dMdx = dMdm[:,:self.num_states]
		dMdu = dMdm[:,self.num_states:]
		dMds = dMds[:,:self.num_states,:self.num_states]
		dSdx = dSdm[:,:,:self.num_states]
		dSdu = dSdm[:,:,self.num_states:]
		dSds = dSds[:,:,:self.num_states,:self.num_states]
		#dVdx = dVdm[:self.num_states,:,:self.num_states]
		#dVdu = dVdm[:self.num_states,:,self.num_states:]
		#dVds = dVds[:self.num_states,:,:self.num_states,:self.num_states]

		if not cross_cov:
			return M, S, dMdx, dMds, dMdu, dSdx, dSds, dSdu
		return M, S, V, dMdx, dMds, dMdu, dSdx, dSds, dSdu