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

from .models import FeasibilityModel

def _generate_data (f, bounds, num_sample):
	if isinstance(num_sample, int):
		num_sample = [ num_sample for _ in bounds ]
	X = np.meshgrid(*[np.linspace(*b, n) for b,n in zip(bounds, num_sample)])
	X = np.vstack([ x.flatten() for x in X ]).T
	Y = np.array([ f(x) for x in X ])
	return X, Y

def sample_optimisation (f, bounds, min_iter=1, max_iter=10, num_sample=51,
						 update_bounds_rule=None, converge_rtol=1e-5,
						 converge_atol=1e-3):
	"""
	Find bounds and feasible region through exhaustive sampling
	"""
	X, Y  = _generate_data( f, bounds, num_sample )
	model = FeasibilityModel( X, Y )

	if not update_bounds_rule is None:
		for i in range( max_iter ):
			dic   = {'X':X, 'Y':Y, 'model':model, 'i':i}
			old_bounds   = bounds.copy()
			bounds, X, Y = update_bounds_rule(bounds, **dic)
			X, Y  = _generate_data( f, bounds, num_sample )
			model = FeasibilityModel( X, Y )

			# Convergence test
			if i+1 >= min_iter:
				if np.allclose(bounds, old_bounds, converge_rtol, converge_atol):
					break

	return model, bounds