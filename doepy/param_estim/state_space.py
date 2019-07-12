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

from scipy.optimize import least_squares as _least_squares
from scipy.optimize import differential_evolution as _differential_evolution

from .loss_func import Residual

class StateSpaceParamEstim:
	def __init__ (self, model, X0, U):
		self.model = model

		if not isinstance(U, (list,tuple)):
			U  = [U]
		self.U = U

		if not isinstance(X0, (list,tuple)):
			X0  = [X0]
		self.X0 = X0

		assert len(self.X0) == len(self.U), 'X0 and U different lengths'

	def __call__ (self, p, T=None):
		self.model.p_mean = p
		Z = [ self.model.predict(x0,u,T)[1] for x0,u in zip(self.X0, self.U) ]
		return np.vstack( Z )


def _noise_var (model, W=None):
	if W is not None:
		return W
	H = model.H         # Observation matrix
	Q = model.x_covar   # Process noise covariance
	R = model.y_covar   # Measurement noise covariance
	return np.einsum('ij,jk,kl->il', H, Q, H.T) + R


def differential_evolution (model, p0, p_bounds, X0, U, Y, W=None, invert_W=True):
	W    = _noise_var(model, W)
	loss = Residual(Y, W, invert_W=invert_W)
	pred = StateSpaceParamEstim(model, X0, U)
	obj  = lambda p: loss.sum_lsq_loss( pred(p) )
	res  = _differential_evolution(obj, p_bounds)
	return res


def least_squares (model, p0, p_bounds, X0, U, Y, W=None, invert_W=True):
	W    = _noise_var(model, W)
	loss = Residual(Y, W, invert_W=invert_W)
	pred = StateSpaceParamEstim(model, X0, U)
	obj  = lambda p: loss.sqrt_lsq_loss( pred(p) )
	res  = _least_squares(obj, p0, bounds=p_bounds.T.tolist())
	return res


def least_squares_grad (model, p0, p_bounds, X0, U, Y, W, invert_W=True):
	"""
	We have model gradients with respect to model parameters
	"""
	raise NotImplementedError('Not yet implemented')
