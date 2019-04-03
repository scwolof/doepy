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
import warnings

from ..utils import assert_not_none
from ..linalg import squaremul

"""
Model:
    x[k+1] = f(x[k],u[k],p) + w[k]
      y[k] = H * x[k] + v[k]
    with w, v noise terms.

Gradients:
    dXdx[k+1] = d f(x[k],u[k],p) / d x[k]
    dXdp[k+1] = d f(x[k],u[k],p) / d p

Parameters p have been fitted to noisy data set Y.
We compute a Laplace approximation of the parameter covariance.
"""

def state_space_noise_var (model):
	H = model.H         # Observation matrix
	Q = model.x_covar   # Process noise covariance
	R = model.y_covar   # Measurement noise covariance
	return squaremul(H, Q) + R

def state_space_compute_dZdp (model, Us):
	if isinstance(Us, np.ndarray):
		Us = [Us]
	dZdp = np.vstack([ _state_space_compute_dZdp(model, U) for U in Us ])
	return dZdp

def _state_space_compute_dZdp (model, U):
	assert_not_none(model.p_mean, '%s:p_mean'%model.name)
	model.p_covar = np.zeros(( model.num_param, model.num_param ))

	N    = U.shape[0]
	X    = np.zeros(( N+1, model.num_states ))
	X[0] = model.x0
	dZdp = np.zeros(( N+1, model.num_meas, model.num_param ))
	dacc = np.zeros(( model.num_states, model.num_param ))

	def func (x, u):
		Sk = np.zeros(( model.num_states, model.num_states ))
		X, _, do = model.predict_x_dist(x, Sk, u, grad=True)
		return X, do.dMdx, do.dMdp

	for n in range(N):
		X[n+1], dxdx, dxdp = func(X[n], U[n])
		dacc      = dxdp + np.matmul(dxdx, dacc)
		dZdp[n+1] = np.matmul(model.H, dacc)
	return dZdp[:-1]

