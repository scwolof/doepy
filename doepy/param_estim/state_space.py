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
from scipy.optimize import least_squares, differential_evolution

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

	def __call__ (self, p):
		self.model.p_mean = p
		Z = [ self.model.predict(x0,u)[1] for x0,u in zip(self.X0, self.U) ]
		return np.vstack( Z )


def _differential_evolution (model, p0, p_bounds, X0, U, Y, y_covar):
	loss = Residual(Y, y_covar, invert_W=True)
	pred = StateSpaceParamEstim(model, X0, U)
	
	obj = lambda p: loss.sum_lsq_loss( pred(p) )
	res = differential_evolution(obj, p_bounds)	
	return res


def _least_squares (model, p0, p_bounds, X0, U, Y, y_covar):
	loss = Residual(Y, y_covar, invert_W=True)
	pred = StateSpaceParamEstim(model, X0, U)
	
	obj = lambda p: loss.sqrt_lsq_loss( pred(p) )
	res = least_squares(obj, p0, bounds=p_bounds.T.tolist())	
	return res