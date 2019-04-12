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

from ...utils import assert_not_none
from ...approximate_inference import taylor_moment_match

class dtNonLinearModel (dtModel):
	def __init__ (self, candidate_model):
		"""
		Transition function f is differentiable:
		    g, do = f( x, u, p, grad=True )
		    x   [ E ]
		    u   [ D ]
		    g   [ E ]
		    do  DerivativeObject

		    WARNING: NOT PROPERLY TESTED WITHOUT HESSIAN INFORMATION
		"""
		super().__init__(candidate_model)

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		# Input mean and covariance
		input_mean, input_cov = self.get_input_mean_and_cov(xk, Sk, u)

		M, do = self.f( *input_mean, grad=True )

		# gradient d f / d input_mean
		dMdm = ( do.dMdx, do.dMdu )
		if self.num_param > 0:
			dMdm += ( do.dMdp, )
		dMdm = np.concatenate( dMdm, axis=1)

		ddM = None
		if grad:
			# Hessian
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

		S, V, do = taylor_moment_match(input_cov, dMdm, ddM, grad)

		S  += self.x_covar
		V   = V[:self.num_states]
		ret = (M, S, V) if cross_cov else (M, S)
		if not grad:
			return ret
		do = self.get_latent_state_derivatives(do)
		return ret+(do,)
