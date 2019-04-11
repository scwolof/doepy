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

from . import DerivativeObject

def taylor_moment_match (s, dMdm, d2Mdm2=None, grad=False):
	E, D = dMdm.shape
	V = np.matmul(s, dMdm.T)
	S = np.matmul(dMdm, V)

	if not grad:
		return S, V

	if d2Mdm2 is None:
		d2Mdm2 = np.zeros((E,D,D))
		# TODO - more clever second derivative approximation?
	
	# Gradients
	do = DerivativeObject(D, E)
	do.dMdx = dMdm.copy()
	do.dVdx = np.einsum('ab,cbd->acd', s, d2Mdm2)
	dsds    = np.eye(D)[:,None,:,None] * np.eye(D)[None,:,None,:]
	do.dVds = np.einsum('abcd,eb->aecd', dsds, dMdm)
	
	for i in range(E):
		do.dSdx[i,i] = 2*np.einsum('bc,bd,d->c',d2Mdm2[i],s,dMdm[i])
		do.dSds[i,i] = dMdm[i,:][:,None] * dMdm[i]
		for j in range(i+1,E):
			do.dSdx[i,j] += np.einsum('bc,bd,d->c', d2Mdm2[i], s, dMdm[j])
			do.dSdx[i,j] += np.einsum('b,bd,dc->c', dMdm[i], s, d2Mdm2[j])
			do.dSdx[j,i]  = do.dSdx[i,j]
			do.dSds[i,j]  = dMdm[i,:][:,None] * dMdm[j]
			do.dSds[j,i]  = do.dSds[i,j].T
			
	return S, V, do




