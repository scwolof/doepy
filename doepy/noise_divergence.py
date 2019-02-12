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

"""
Divergence between observed noise distribution of the filtered/smoothed
posterior and the expected noise distribution.
"""

def KL_P0Pw (Y, Z, S):
	r"""
	Kullback-Leibler divergence between N(0,I) and normalised noise distribution

	Y  [N x E]
	Z  [M x N x E]
	S  [M x N x E x E]
	"""
	mw, Sw = _normalise(Y, Z, S) # [M x E], [M x E x E]

	M, E = mw.shape
	D    = np.zeros( M )
	iSw  = np.linalg.inv( Sw )
	for i in range( M ):
		miSm = np.matmul( mw[i], np.matmul(iSw[i], mw[i]) )
		D[i] = np.trace(iSw[i]) + miSm - E + np.log(np.linalg.det(Sw[i]))
	return 0.5 * D


def KL_PwP0 (Y, Z, S):
	r"""
	Kullback-Leibler divergence between normalised noise distribution and N(0,I)

	Y  [N x E]
	Z  [M x N x E]
	S  [M x N x E x E]
	"""
	mw, Sw = _normalise(Y, Z, S) # [M x E], [M x E x E]

	M, E = mw.shape
	D = np.zeros( M )

	for i in range( M ):
		mm   = np.sum( mw[i]*mw[i] )
		D[i] = np.trace(Sw[i]) + mm - E - np.log(np.linalg.det(Sw[i]))
	return 0.5 * D


def _normalise (Y, Z, S):
	r"""
	Compute normalised random variable
		w_k \sim L_k^T ( z_k - y_k ) 
	where
		Y   [N x E ]		Observations y_k, k = 1,...,N
		Z   [M x N x E]		Mean predictions z_k for all M models
		S   [M x N x E x E] Covariances S_k, with Cholesky decomposition S=LL^T
	"""
	W  = np.zeros(Z.shape)
	iL = np.linalg.inv( np.linalg.cholesky(S) )
	for i in range( len(Z) ):
	    for j in range( len(Y) ):
	        W[i,j] = np.matmul( Z[i,j] - Y[j], iL[i,j].T )

	# Divergence from N(0,I)-distribution
	mean = np.mean( W, axis=1 )
	tmp  = W - mean[:,None,:]
	var  = 1./(W.shape[1]-1) * np.sum( tmp[:,:,:,None]*tmp[:,:,None,:], axis=1 )

	return mean, var