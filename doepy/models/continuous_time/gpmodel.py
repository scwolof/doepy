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
	def _ode_moment_match (self, x, S, u, grad=False):
		input_mean, _ = self.get_input_mean_and_cov(x, S, u, concatenate=True)

		if self.transform:
			assert_not_none(self.output_transform, 'output_transform')
			assert_not_none(self.input_transform, 'input_transform')
			input_mean = self.input_transform( input_mean )

		E, D, P = self.num_states, self.num_inputs, self.num_param
		dM, dS  = np.zeros( E ), np.zeros(( E, E ))
		for e, gp in enumerate(self.gps):
			tmp     = gp.predict_noiseless(input_mean[None,:])
			dM[e]   = tmp[0][0,0]
			dS[e,e] = tmp[1][0,0]
		    
		do = Derivatives(E, num_states=E+D+P)

		if grad:
			do.dMdx, do.dSdx = d_pred_d_x(self.gps, input_mean, diag=True)
			do.dMdxx = d2_m_d_x2(self.gps, input_mean)
		else:
			do.dMdx  = d_pred_d_x(self.gps, input_mean, mean_only=True)

		# Transform back
		if self.transform:
			qt,qz    = self.input_transform.q, self.output_transform.q
			dM       = self.output_transform(dM, back=True)
			dS       = self.output_transform.cov(dS, back=True)
			qtqt     = qt[:,None] * qt[None,:]
			qzqz     = qz[:,None] * qz[None,:]
			do.dMdx *= qz[:,None] / qt[None,:]
			if grad:
				do.dSdx  *= qzqz[:,:,None] / qt[None,None,:]
				do.dMdxx *= qz[:,None,None] / qtqt[None,:,:]

		do = self.get_latent_state_derivatives(do, cross_cov=False, hessian=grad)

		dS += self.x_covar + np.matmul(do.dMdx, S) + np.matmul(S, do.dMdx.T)
		if not grad:
			return dM, dS, do

		I = np.eye(self.num_states)[:,None,:,None] \
		    * np.eye(self.num_states)[None,:,None,:]
		dSdx     = np.einsum("kj,ijn->kin", S, do.dMdxx)
		do.dSdx += dSdx + np.transpose(dSdx, [1,0,2])
		dSdu     = np.einsum("kj,ijn->kin", S, do.dMdxu)
		do.dSdu += dSdu + np.transpose(dSdu, [1,0,2])
		dSds     = np.einsum('ij,jkmn->ikmn', do.dMdx, I)
		do.dSds += dSds + np.transpose(dSds, [1,0,3,2])
		if self.num_param > 0:
			dSdp     = np.einsum("kj,ijn->kin", S, do.dMdxp)
			do.dSdp += dSdp + np.transpose(dSdp, [1,0,2])

		return dM, dS, do