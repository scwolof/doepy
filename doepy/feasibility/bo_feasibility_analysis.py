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

from .models import GPFeasibilityModel
from ..optimize import multistart_points, multistart_fmin_l_bfgs_b


def bayesian_optimisation (f, acq_func, X, Y, bounds, min_iter=20, max_iter=50,
                           retrain_hyp_test=None, update_bounds_rule=None,
                           multistart_N_test=500, multistart_N_run=10,
                           converge_rtol=1e-5, converge_atol=1e-3):
	"""
	We want to _maximise_ the acquisition function 
	"""
	model    = GPFeasibilityModel( X, Y.reshape((X.shape[0],1)) )
	max_iter = np.max(( min_iter, max_iter ))
	retrain_hyp = True
	convergence = False

	for i in range( max_iter ):
		# Initial points
		func   = lambda x: -acq_func(x, model)
		X_init = multistart_points( func, bounds, N_test=multistart_N_test, \
		                            N_return=multistart_N_run )

		# Multistart optimisation
		func     = lambda x: [-e[0] for e in acq_func(x, model, grad=True)]
		xbest, _ = multistart_fmin_l_bfgs_b(func, X_init, bounds)
		y = f(xbest)
		y = [ y ] if isinstance(y, float) else y
		Y = np.concatenate(( Y, y ))
		X = np.vstack(( X, xbest ))
		
		# Update bounds
		if not update_bounds_rule is None:
			model = GPFeasibilityModel( X, Y.reshape((X.shape[0],1)), model[:] )
			dic   = {'X':X, 'Y':Y, 'model':model, 'i':i}
			old_bounds   = bounds.copy()
			bounds, X, Y = update_bounds_rule( bounds, **dic )

			# Convergence test
			if i+1 >= min_iter:
				if np.allclose(bounds, old_bounds, converge_rtol, converge_atol):
					convergence = True

		model = GPFeasibilityModel( X, Y.reshape((X.shape[0],1)), model[:] )

		# Update model (with hyperparameter training)
		if retrain_hyp_test is not None:
			if retrain_hyp_test( i ) or i == max_iter-1:
				model.optimize()

		if convergence:
			break

	return model, bounds
