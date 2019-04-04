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
from scipy.stats import multivariate_normal as mvn

"""
M - number of models
N - number of test points
E - number of target dimensions

Y - observations
Z - mean
S - covariance
"""

def relative_likelihood (Y, Z, S):
	"""
	Normalised log marginal likelihood:
	   nlml_i = exp( prob_i ) / sum_j( exp( prob_j ) )

	Y  [N x E]
	Z  [M x N x E]
	S  [M x N x E x E]
	"""
	loglikf = lambda Y,Z,S: [ mvn(z,s2).logpdf(y) for y,z,s2 in zip(Y, Z, S) ]
	logliks = [ np.sum(loglikf(Y,z,s2)) for z,s2 in zip(Z, S) ]

	# Use min( max() ) for numerical stability
	# -- Does not affect results significantly
	diff    = lambda p1, p2: np.min(( np.max((p2-p1, -1000)), 100 ))
	rellikf = lambda p1, p2: np.exp( diff(p1, p2) )
	relliks = [ 1./np.sum([rellikf(p1,p2) for p2 in logliks]) for p1 in logliks]
	return np.array( relliks )
