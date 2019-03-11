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

def find_active_dims (f, xbs, ubs, num_points=5, threshold=1e-10):
	"""
	xbs - x bounds
	ubs - u bounds
	"""
	D = xbs.shape[0]
	B = ubs.shape[0]
	
	assert xbs.shape == (D, 2)
	X = [ np.linspace(*xb, num_points) for xb in xbs ]
	X = np.vstack([x for x in X]).T
	
	assert ubs.shape == (B, 2)
	U = [ np.linspace(*ub, num_points) for ub in ubs ]
	U = np.vstack([u for u in U]).T
	
	# N-1 model evaluations
	Y = np.array([ f(X[n],U[n]) for n in range(num_points-1) ])
	
	T  = np.c_[X, U]
	Di = np.zeros((D+B, D))
	# (N-1)*(D+B) model evaluations
	for n in range(num_points-1):
		for d1 in range(D+B):
			t       = T[n].copy()
			t [d1]  = T[n+1,d1]
			y       = f(t[:D], t[D:])
			Y, T    = np.vstack((Y, y)), np.vstack((T, t))
			Di[d1] += np.abs(Y[n] - y)
	
	C = Di.T >= threshold
	# Total (N-1)*(1+D+B) model evaluations
	return [ [i for i in np.arange(D+B) if C[d,i]] for d in range(D) ], Y, T