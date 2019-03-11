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

from .find_active_dims import find_active_dims

def generate_training_data (f, x_bounds, u_bounds, active_dims=None,
                   num_data_points_per_num_dim_combo = [101, 21, 9, 5, 4, 3]):
	Ns = num_data_points_per_num_dim_combo

	if active_dims is None:
		active_dims, Y, T = find_active_dims(f, x_bounds, u_bounds)

	num_meas = len( active_dims )
	X, U, Y  = ([None]*len(active_dims) for _ in range(3))
	D, B     = x_bounds.shape[0], u_bounds.shape[0]
	t_bounds = np.vstack(( x_bounds, u_bounds ))

	for i in range( num_meas ):
		ad = active_dims[i]
		n  = len( ad )
		N  = Ns[-1] if n > len(Ns) else Ns[n-1]
		t  = []
		for j in range( D+B ):
			if j in ad:
				t.append( np.linspace(*t_bounds[j],N) )
			else:
				t.append([0])
		t  = np.meshgrid(*t)
		t  = np.vstack(( ti.flatten() for ti in t )).T
		X[i], U[i] = t[:,:D], t[:,D:]

	"""
	TODO:
	Combine non-interfering inputs to reduce the number of times the model
	needs to be evaluated
	"""

	for i in range( num_meas ):
		Y[i] = np.array([ f(x,u)[i] for x,u in zip(X[i],U[i]) ])

	return X, U, Y