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

from .state_space import state_space_noise_var, state_space_compute_dZdp
from ..linalg import squaremul

from pdb import set_trace as st

def _safe_p_covar ( inv_p_covar, diag_min_ratio=0.5, diag_min_thresh_ratio=0.1, 
	off_diag_abs_max_ratio=0.99, warning_msg=False ):
	
	if warning_msg:
		# For printing warning messages
		def warn (msg):
			warnings.warn('safe_p_covar: %s'%msg)
	
	if not np.allclose(inv_p_covar, inv_p_covar.T):
		warn('Forcing symmetric matrix')
	inv_p_covar = 0.5 * (inv_p_covar+ inv_p_covar.T)
	
	num   = inv_p_covar.shape[0]
	diag  = np.diag( inv_p_covar )

	""" Minimum value on diagonal """
	maxdiag = np.max(diag)
	if maxdiag <= 0.:
		raise ValueError('Maximum diagonal value must be strictly positive')
	mindiag = diag_min_ratio * maxdiag
	thrdiag = diag_min_thresh_ratio * maxdiag
	for i in range( num ):
		if thrdiag < inv_p_covar[i,i] < mindiag:
			"""
			Lower bound on diagonal elements > diag_min_thresh
			"""
			inv_p_covar[i,i] = mindiag
			if warning_msg:
				warn('Setting [%d,%d] to mindiag (%f)'%(i,i,mindiag))

		elif inv_p_covar[i,i] <= thrdiag:
			"""
			Bound on diagonal elements <= diag_min_thresh
			
			Problem with model parameters which there aren't enough data for.
			Can either set the corresponding variance to infinite (which is 
			theoretically more correct) or to zero (which is numerically more 
			stable).
			--> We're going with setting it to zero ( 1. / np.inf )
			Equivalent to fixing a parameter that appears unidentifiable.
			"""
			inv_p_covar[i,i] = np.inf
			if warning_msg:
				warn('Setting [%d,%d] to inf'%(i,i))

	""" Make sure off-diag terms are not (absolutely) larger than diag terms """
	for i in range(num):
		for j in range(i+1, num):
			# Smallest diagonal term
			dterm  = np.min(( inv_p_covar[i,i], inv_p_covar[j,j] ))
			thresh = off_diag_abs_max_ratio * dterm
			# Off-diagonal term
			oterm  = inv_p_covar[i,j]
			# Absolute value
			absval = np.abs( oterm )
			if not absval > 0:
				continue
			ratio  = thresh / absval
			
			if ratio < 1:
				inv_p_covar[i,j] = oterm * ratio
				inv_p_covar[j,i] = inv_p_covar[i,j]
				if warning_msg:
					warn('Reducing [%d,%d] by ratio %f'%(i,j,ratio))
				
	return inv_p_covar


def _laplace ( model, Us, noise_var, compute_dZdp, \
			   safe_p_covar=_safe_p_covar ):
	try:
		iW = np.linalg.inv( noise_var(model) )
	except:
		W = np.eye( model.num_meas )
		warnings.warn('Could not invert weight matrix. Setting W = I.')

	dZdp    = compute_dZdp(model, Us)
	i_p_cov = np.sum([ squaremul(dz.T,iW) for dz in dZdp ], axis=0)
	if safe_p_covar is not None:
		i_p_cov = safe_p_covar(i_p_cov)

	p_covar = np.linalg.inv( i_p_cov )
	# Force symmetric
	p_covar = 0.5 * (p_covar + p_covar.T)
	return p_covar


def state_space_laplace (model, Us, **kwargs):
	noise_var    = state_space_noise_var
	compute_dZdp = state_space_compute_dZdp
	return _laplace(model, Us, noise_var, compute_dZdp, **kwargs)