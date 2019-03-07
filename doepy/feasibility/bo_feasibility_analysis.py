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

from GPy.models import GPRegression
from GPy.kern import Matern52

from ..optimize import multistart_points, multistart_fmin_l_bfgs_b

def train_feasibility_gpmodel (X, Y, hyp=None, retrain_hyp=True, \
                               noise_var_trainable=False, noise_var=1e-4):
	feas_model = GPRegression(X, Y[:,None], Matern52(X.shape[1], ARD=True))
	# Set hyperparameters if provided
	if hyp is not None:
		feas_model.update_model(False)
		feas_model.initialize_parameter()
		feas_model[:] = hyp
		noise_var     = feas_model.likelihood.variance[0]
		feas_model.update_model(True)
	else:
		retrain_hyp = True
	# Fix noise variance, if not trainable
	if not noise_var_trainable:
		feas_model.Gaussian_noise.variance.constrain_fixed(noise_var)
	# Optimise hyperparameters if necessary
	if retrain_hyp:
		feas_model.optimize()
	return feas_model


def bayesian_optimisation (f, acq_func, feas_model, bounds, max_iter=25,\
                           retrain_hyp_test=None, update_bounds_rule=None,
                           multistart_N_test=500, multistart_N_run=10):
	"""
	We assume we want to _maximise_ the acquisition function 
	"""
	X, Y = feas_model.X, feas_model.Y[:,0]
	retrain_hyp = True

	for i in range( max_iter ):
		# Initial points
		func   = lambda x: -acq_func(x, feas_model)
		X_init = multistart_points( func, bounds, N_test=multistart_N_test, \
		                            N_return=multistart_N_run )

		# Multistart optimisation
		func     = lambda x: [-e[0] for e in acq_func(x, feas_model, grad=True)]
		xbest, _ = multistart_fmin_l_bfgs_b(func, X_init, bounds)
		y = f(xbest)
		y = [ y ] if isinstance(y, float) else y
		Y = np.concatenate(( Y, y ))
		X = np.vstack(( X, xbest ))
		
		# Update bounds
		if not update_bounds_rule is None:
			feas_model = train_feasibility_gpmodel( X, Y, hyp=feas_model[:], \
			                                        retrain_hyp=False )
			dic = {'X':X, 'Y':Y, 'model':feas_model, 'i':i}
			bounds, X, Y = update_bounds_rule( bounds, **dic )

		# Update model (with hyperparameter training)
		if retrain_hyp_test is not None:
			retrain_hyp = retrain_hyp_test( i ) or i == max_iter-1
		feas_model = train_feasibility_gpmodel( X, Y, hyp=feas_model[:], \
		                                        retrain_hyp=retrain_hyp )

	return feas_model, bounds
