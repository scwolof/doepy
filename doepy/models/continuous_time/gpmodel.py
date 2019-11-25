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

from .model import ctModel
from ..gpmodel import GPModel
from ..derivatives import LatentStateDerivatives

from ...utils import assert_not_none
from ...training import generate_training_data
from ...derivatives import Derivatives
from ...constraints import ConstantMeanStateConstraint
from ...approximate_inference.gp_derivatives import d_pred_d_x, d2_m_d_x2

default_noise_var = 1e-5

class ctGPModel (ctModel, GPModel):
	def __init__ (self, candidate_model):
		"""
		We assume we do not have gradient information for f
		We place a GP prior on f
		"""
		ctModel.__init__(self, candidate_model)
		GPModel.__init__(self, transform=candidate_model.transform)

		assert_not_none(self.u_bounds,'u_bounds')

		assert_not_none(candidate_model.x_bounds,'candidate:x_bounds')
		self.x_bounds      = candidate_model.x_bounds
		self.x_constraints = None

		self.delta_transition = False

	"""
	State prediction
	"""
	def _ode_moment_match (self, M, S, grad=False, exact_mean=False):
		if self.transform:
			assert_not_none(self.output_transform, 'output_transform')
			assert_not_none(self.input_transform, 'input_transform')
			input_mean = self.input_transform( M )
			
		Dx, Du, Dp = self.num_states, self.num_inputs, self.num_param
		E = self.num_states + self.num_inputs + self.num_param
		dm, ds  = np.zeros( Dx ), np.zeros(( Dx, Dx ))
			
		dMdx, dSdx = d_pred_d_x(self.gps, input_mean, diag=True)
		if grad:
			dMdxx = d2_m_d_x2(self.gps, input_mean)
		
		if exact_mean:
			input_mean = (M[:Dx], M[Dx:(Dx+Du)])
			if Dp > 0:
				input_mean += ( M[(Dx+Du):], )
			dm = self.f(*input_mean)
		else:
			for e, gp in enumerate(self.gps):
				tmp     = gp.predict_noiseless(input_mean[None,:])
				dm[e]   = tmp[0][0,0]
				ds[e,e] = tmp[1][0,0]

		# Transform back
		if self.transform:
			qt,qz = self.input_transform.q, self.output_transform.q
			### 
			if not exact_mean:
				dm = self.output_transform(dm, back=True)
				ds = self.output_transform.cov(ds, back=True)
			###
			qtqt  = qt[:,None] * qt[None,:]
			qzqz  = qz[:,None] * qz[None,:]
			dMdx *= qz[:,None] / qt[None,:]
			if grad:
				dSdx  *= qzqz[:,:,None] / qt[None,None,:]
				dMdxx *= qz[:,None,None] / qtqt[None,:,:]

		do = Derivatives(E, num_states=E)
		do.dMdx = np.zeros((E, E))
		do.dMdx[:Dx] = dMdx
				
		dM, dS  = np.zeros( E ), np.zeros(( E, E ))
		dM[:Dx] = dm
		dS[:Dx,:Dx] = ds + self.x_covar
		dS += np.matmul(do.dMdx, S) + np.matmul(S, do.dMdx.T)
		if not grad:
			return dM, dS, do
		
		do.dSdx  = np.zeros((E, E, E))
		do.dMdxx = np.zeros((E, E, E))
		do.dSdx[:Dx,:Dx] = dSdx
		do.dMdxx[:Dx] = dMdxx
			
		I = np.eye(E)[:,None,:,None] * np.eye(E)[None,:,None,:]
		dSdx     = np.einsum("kj,ijn->kin", S, do.dMdxx)
		do.dSdx += dSdx + np.transpose(dSdx, [1,0,2])
		dSds     = np.einsum('ij,jkmn->ikmn',do.dMdx,I)
		do.dSds += dSds + np.transpose(dSds, [1,0,3,2])

		return dM, dS, do