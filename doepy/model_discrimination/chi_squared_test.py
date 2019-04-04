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
from scipy.stats import chi2

from ..utils import assert_is_shape

"""
Chi-squared adequacy test

M - number of models
N - number of test points
E - number of target dimensions

Y - observations
Z - mean
S - covariance
D - number of model parameters
"""
def chi2_adequacy_test (Y, Z, S, D):
	"""

	Y  [N x E]
	Z  [M x N x E]
	S  [M x N x E x E]
	D  [M]
	"""
	num_models = Z.shape[0]

	D = np.asarray(D)
	assert_is_shape(D, (num_models,))

	# Squared Mahalanobis distance
	mahaf1 = lambda d, is2: np.sum( d * np.matmul(is2, d) )
	# Mahalanobis distances for all test points
	mahaf2 = lambda Y, Z, iS: [ mahaf1(y-z, is2) for y,z,is2 in zip(Y,Z,iS) ]

	iS   = np.linalg.inv(S)
	maha = [ np.sum(mahaf2(Y,z,is2)) for z,is2 in zip(Z, iS)]
	maha = np.array( maha )

	# Degrees of freedom
	dof = Y.shape[0]*Y.shape[1] - np.asarray(D)

	if np.any(dof <= 0):
		raise RuntimeWarning('Degrees of freedom not greater than zero.')
	return 1. - chi2.cdf(maha, dof)