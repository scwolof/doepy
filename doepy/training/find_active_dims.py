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
	# Input dimensions (dim_x, dim_u, dim_p)
	Ds = [ b.shape[0] for b in bounds_list ]
	# Dimensions indices (0, dim_x, dim_x+dim_u, dim_x+dim_u+dim_p)
	Dss = np.cumsum([0] + Ds).tolist()
	# Total number of input dimensions dim_x+dim_u+dim_p
	Dtot = np.sum( Ds )
	# Concatenated dimension bounds
	bnds = np.vstack(bounds_list)
	assert bnds.shape == (Dtot, 2), 'Illegal matrix shape'

	def stack_linspace (bounds):
		return np.vstack([ np.linspace(*b, num_points+1) for b in bounds ])
	X = np.vstack([ stack_linspace(b) for b in bounds_list ]).T

	def separate (t):
		# Separate into individual inputs (x, u, p)
		return [ t[di:dj] for di,dj in zip(Dss, Dss[1:]) ]

	def difference (Y, y):
		E  = len( y )
		df = np.zeros(E)
		for n in range(E):
			if np.isnan(Y[n]) and np.isnan(y[n]):
				df[n] = 0.
			elif np.isnan(Y[n]) or np.isnan(y[n]):
				df[n] = 10 * threshold
			else:
				df[n] = np.abs(Y[n]-y[n])
		return df

	def compare (differences):
		D,E = differences.shape
		C   = np.zeros((E,D),bool)
		for d in range(D):
			for e in range(E):
				C[e,d] = differences[d,e] >= threshold
		return C

	C = np.zeros((num_outputs, Dtot), bool) 

	# num_points-1 model evaluations
	Y = np.array([ f(*separate(x)) for x in X[:-1] ])
	E = Y.shape[1]

	Di = np.zeros((Dtot, E))
	# At most (num_points-1)*Dtot model evaluations
	for n in range( num_points ):
		for d in range( Dtot ):
			if np.all( C[:,d] ):
				# Input active for all dimensions, no need to test
				continue
			x      = X[n].copy()
			x[d]   = X[n+1,d]
			y      = f(*separate(x))
			Y, X   = np.vstack((Y, y)), np.vstack((X, x))
			Di[d] += difference(Y[n], y)
		C = compare(Di)

	active_dims = [ [i for i in np.arange(Dtot) if C[e,i]] for e in range(E) ]
	# At most (num_points-1)*(1+Dtot) model evaluations
	return (active_dims, X, Y) if return_data else active_dims