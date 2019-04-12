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

from GPy.kern import RBF

from . import Derivatives

def exact_moment_match (gps, mu, s2, grad=False, independent=False):
	"""
	Approximate inference with uncertain input x ~ N(mu, s2)
	for function with GP prior given by gps.

	- Only exact for GPs with RBF kernels
	- Can handle GPs with RBF kernels with different active dimensions.

	Inputs:
	    gps   [ E ]      list, GPs (one for each independent target dimension)
	    mu    [ D ]      numpy array, input mean 
	    s2    [ D x D ]  numpy array, input covariance
	(optional)
	    grad             return gradients of outputs wrt inputs
	    independent      target dimensions independent
	                     do not compute cross-covariance terms

	Outputs:
	    M     [ E ]      mean E_{x,f}[ f(x) ]
	    S     [ E x E ]  covariance V_{x,f}[ f(x) ]
	    V     [ D x E ]  input-output covariance cov_{x,f}[ x, f(x) ]
	    if grad, return DerivativeObject with:
	    dMdx  [ E x D ]
	    dMds  [ E x D x D ]
	    dSdx  [ E x E x D ]
	    dSds  [ E x E x D x D ]
	    dVdx  [ D x E x D ]
	    dVds  [ D x E x D x D ]

	Code largely based on PILCO (Copyright (C) 2008-2013 by Marc Deisenroth, 
	Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen) written in MATLAB.
	"""
	assert isinstance(gps, list) and not gps == []

	# Check that GPs have RBF kernels
	for e, gp in enumerate(gps):
		assert isinstance(gp.kern, RBF),\
		    'GP %d does not have RBF kernel: %s'%(e, gp.kern)

	# Memory allocation
	D = len( mu )        # Number of input dimensions
	E = len( gps )       # Number of target dimensions
	
	M = np.zeros( (E) )
	S = np.zeros( (E, E) )
	V = np.zeros( (D, E) )

	if grad:
		do = DerivativeObject(D, E)

	lengthscales    = 100 * np.ones((E, D))
	all_dims_active = True
	for e in range( E ):
		I = gps[e].kern.active_dims
		if not len( I ) == D:
			all_dims_active = False
		lengthscales[e,I] = np.asarray(gps[e].kern.lengthscale)

	logk = []
	for e in range( E ):
		if e == 0 or not all_dims_active:
			# Full or sparse GP ?
			I   = gps[e].kern.active_dims
			N   = gps[e].posterior.woodbury_vector.shape[0]
			inp = np.zeros(( N, D ))
			# Centralise training inputs
			inp[:,I] = gps[e]._predictive_variable[:,I] - mu[None,I]

		# First, some useful intermediate terms
		beta = gps[e].posterior.woodbury_vector.flatten()
		leng = lengthscales[e]
		rho2 = np.array(gps[e].kern.variance)[0]
		
		iL    = np.diag(1. / leng**2 )
		is2L  = np.linalg.inv( s2 + np.diag(leng**2) )
		is2LX = np.matmul(inp, is2L)
		q     = np.exp(-0.5 * np.sum(inp * is2LX, axis=1))
		bq    = beta * q
		s2LI  = np.matmul(s2, iL) + np.eye(D)
		c     = rho2 / np.sqrt(np.linalg.det( s2LI ))
		
		sumbq = np.sum( bq )
		M[e]  = c * sumbq
		if grad:
			dbqdm   = bq[:,None] * is2LX
			dbqds   = 0.5 * bq[:,None,None]*is2LX[:,:,None]*is2LX[:,None,:]
			dcds    = -0.5 * np.matmul( np.linalg.inv(s2LI).T, iL )
			do.dMdx[e] = c * np.sum( dbqdm, axis=0 )
			do.dMds[e] = c * ( np.sum( dbqds, axis=0 ) + dcds * sumbq )

		is2LXs2 = np.matmul(is2LX, s2.T)
		V[:,e]  = c * np.sum( bq[:,None] * is2LXs2, axis=0 )
		if grad:
			do.dVdx[:,e] = 2 * np.matmul(s2, do.dMds[e])
			do.dVds[:,e] =  dcds * V[:,e,None,None]
			s2is2L = np.matmul(s2, is2L)
			for d1 in range(D):
				dis2LXs2ds = - is2LX[:,None,d1,None] * s2is2L[None,:]
				dis2LXs2ds[:,d1] += is2LX
				dsumbqSSds = np.sum( dbqds[:,None,d1] * is2LXs2[:,:,None] \
								+ bq[:,None,None] * dis2LXs2ds, axis=0 )
				do.dVds[:,e,d1] += c * dsumbqSSds
			
		logk.append( np.log(rho2) - 0.5 * np.sum( (inp/leng)**2, axis=1 ) )
		
	# 2) predictive covariance matrix
	# 2a) non-central moments
	for i in range(E):
		if not all_dims_active:
			# Full or sparse GP ?
			I   = gps[i].kern.active_dims
			N   = gps[i].posterior.woodbury_vector.shape[0]
			inp = np.zeros(( N, D ))
			# Centralise training inputs
			inp[:,I] = gps[i]._predictive_variable[:,I] - mu[None,I]

		lengi = 1. / lengthscales[i]**2
		ii    = inp * lengi
		betai = gps[i].posterior.woodbury_vector

		for j in range(i,E):
			if j == i:
				lengj = lengi
				ij    = ii
			elif not all_dims_active:
				# Full or sparse GP ?
				I   = gps[j].kern.active_dims
				N   = gps[j].posterior.woodbury_vector.shape[0]
				inp = np.zeros(( N, D ))
				# Centralise training inputs
				inp[:,I] = gps[j]._predictive_variable[:,I] - mu[None,I]

				lengj = 1. / lengthscales[j]**2
				ij    = inp * lengj

			betaj = gps[j].posterior.woodbury_vector
			
			R     = np.matmul(s2, np.diag(lengi + lengj)) + np.eye(D)
			iR    = np.linalg.inv( R )
			isdR  = 1. / np.sqrt( np.linalg.det(R) )
			iRs2  = np.matmul( iR, s2 )
			zi,zj = np.matmul(ii,iRs2), np.matmul(ij,iRs2)
			i1,j2 = np.sum(zi*ii,1), np.sum(zj*ij,1)
			zRSz  = (i1[:,None] + j2[None,:]) + 2*np.matmul(zi,ij.T)
			Q     = np.exp( (logk[i][:,None] + logk[j]) + 0.5*zRSz )

			A = betai * betaj.T
			if i == j:
				# Incorporate model uncertainty
				A -= gps[i].posterior.woodbury_inv
			A = A * Q
			
			S[i,j] = np.sum( A ) * isdR
			S[j,i] = S[i,j]
			
			if grad:
				zi  = np.matmul(ii, iR)
				zj  = np.matmul(ij, iR)
				r,T = np.zeros(D), np.zeros((D,D))
				for d in range(D):
					B    = (zi[:,[d]] + zj[:,[d]].T ) * A
					r[d] = np.sum(B) * isdR
					T[d, :d+1] = np.sum(np.matmul(zi[:,:d+1].T, B), axis=1)\
								+ np.sum(np.matmul(B, zj[:,:d+1]), axis=0)
					T[:d+1, d] = T[d,:d+1]
				
				r -= M[i]*do.dMdx[j] + M[j]*do.dMdx[i] 
				do.dSdx[i,j], do.dSdx[j,i] = r.copy(), r.copy()
				T  = 0.5 * (isdR * T - S[i,j] * iR*(lengi + lengj)[:,None])
				T -= M[i]*do.dMds[j] + M[j]*do.dMds[i] 
				do.dSds[i,j], do.dSds[j,i] = T.copy(), T.copy()

			if independent:
				break
				
		S[i,i] += np.array(gps[i].kern.variance)[0]
	# 2b) centralise moments
	S -= M[:,None] * M[None,:]

	if grad:
		return M, S, V, do
	return M, S, V, None