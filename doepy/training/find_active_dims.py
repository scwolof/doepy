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

def find_active_dims (f, bounds_list, num_points=5, threshold=1e-10,
                      return_data=False):
	"""
	bounds_list = [ xbs, ubs, (pbs) ]
	xbs - latent state bounds
	ubs - control input bounds
	pbs - model parameter bounds (optional)

	The bounds lists have to be in the order that the parameters are given
	as inputs to the function f, i.e.
	bounds_list = [xbs, ubs]       =>  y = f(x, u)
	bounds_list = [xbs, ubs, pbs]  =>  y = f(x, u, p)
	"""
	Ds   = [ b.shape[0] for b in bounds_list ]
	Dss  = np.cumsum([0] + Ds).tolist()
	Dtot = np.sum( Ds )
	bnds = np.vstack(bounds_list)
	assert bnds.shape[1] == 2

	def stack_linspace (bounds):
		return np.vstack([ np.linspace(*b, num_points) for b in bounds ])
	X = np.vstack([ stack_linspace(b) for b in bounds_list ]).T

	def separate (x):
		return [ x[di:dj] for di,dj in zip(Dss, Dss[1:]) ]
	
	# num_points-1 model evaluations
	Y = np.array([ f(*separate(x)) for x in X[:-1] ])
	E = Y.shape[1]
	
	Di = np.zeros((Dtot, E))
	# (num_points-1)*Dtot model evaluations
	for n in range( num_points-1 ):
		for d in range( Dtot ):
			x      = X[n].copy()
			x[d]   = X[n+1,d]
			y      = f(*separate(x))
			Y, X   = np.vstack((Y, y)), np.vstack((X, x))
			Di[d] += np.abs(Y[n] - y)
	
	C = Di.T >= threshold
	active_dims = [ [i for i in np.arange(Dtot) if C[e,i]] for e in range(E) ]
	# Total (num_points-1)*(1+Dtot) model evaluations
	return (active_dims, X, Y) if return_data else active_dims