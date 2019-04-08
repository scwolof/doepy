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

from . import LatentStateDerivativeObject

from .model import dtModel
from ...utils import assert_not_none
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
		inp = (xk, u)
		if self.num_param > 0:
			assert_not_none(self.p_mean, '%s:p_mean'%self.name)
			assert_not_none(self.p_covar, '%s:p_covar'%self.name)
			inp += (self.p_mean,)

		M, do = self.f( *inp, grad=True )

		if self.hessian:
			assert do.dMdx.ndim == 2, 'Number of test points must be None!'
			x_xu  = np.concatenate(( do.dMdxx, do.dMdxu ), axis=2)
			dMdxu = np.transpose( do.dMdxu, axes=[0,2,1] )
			ux_u  = np.concatenate(( dMdxu, do.dMduu ), axis=2)
			ddM   = np.concatenate(( x_xu, ux_u ), axis=1)
			if self.num_param > 0:
				xup = np.concatenate(( do.dMdxp, do.dMdup ), axis=1 )
				ddM = np.concatenate(( ddM, xup ), axis=2 )
				xup = np.transpose( xup, axes=[0,2,1] )
				xup = np.concatenate(( xup, do.dMdpp ), axis=2 )
				ddM = np.concatenate(( ddM, xup ), axis=1 )
		else:
			ddM = None

		dMdm = np.concatenate(( do.dMdx, do.dMdu ), axis=1)
		dim  = [self.num_states, self.num_inputs]
		Ss   = [Sk, self.u_covar]
		if self.num_param > 0:
			dMdm = np.concatenate(( dMdm, do.dMdp ), axis=1)
			dim += [self.num_param,]
			Ss  += [self.p_covar]
		Dss  = np.cumsum([0,] + dim)
		Snew = np.zeros((Dss[-1], Dss[-1]))
		for i,Si in enumerate( Ss ):
			i1, i2 = Dss[i], Dss[i+1]
			Snew[i1:i2, i1:i2] = Si

		if grad:
			S, V, domm = taylor_moment_match(Snew, dMdm, ddM, True)
		else:
			S, V = taylor_moment_match(Snew, dMdm)

		S  += self.x_covar
		V   = V[:self.num_states]
		ret = (M, S, V) if cross_cov else (M, S)

		if grad:
			do = LatentStateDerivativeObject(self)
			D  = self.num_states
			dn = D + self.num_inputs
			do.dMdx = domm.dMdm[:,:D]
			do.dMdu = domm.dMdm[:,D:dn]
			do.dMds = domm.dMds[:,:D,:D]
			do.dSdx = domm.dSdm[:,:,:D]
			do.dSdu = domm.dSdm[:,:,D:dn]
			do.dSds = domm.dSds[:,:,:D,:D]
			do.dVdx = domm.dVdm[:D,:,:D]
			do.dVdu = domm.dVdm[:D,:,D:dn]
			do.dVds = domm.dVds[:D,:,:D,:D]
			if self.num_param > 0:
				P = self.num_param
				do.dMdp = domm.dMdm[:,-P:]
				do.dSdp = domm.dSdm[:,:,-P:]
				do.dVdp = domm.dVdm[:,:,-P:]
			ret += (do,)
 
		return ret
