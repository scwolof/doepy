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
from scipy.optimize import minimize


def stretch_bounds (bounds, alpha=0.2):
	a = alpha / 2.
	A = np.array([ [1+a, -a], [-a, 1+a] ])
	return np.matmul(bounds, A)


class BoundUpdate:
	def __init__ (self, known_dimensions=[], remove_data=True):
		"""
		known_dimensions - list with indices for dimensions with known bounds
		                   (these bounds should not be modified!)
		"""
		self.known_dim = known_dimensions
		self.rm_data   = remove_data

	def __call__ (self, bounds, **kwargs):
		# Update bounds
		bounds, X, y = self._update_unknown_dims(bounds, **kwargs)

		# Remove data outside box
		if self.rm_data:
			I = [i for i,x in enumerate(X) \
				if  np.all( x >= bounds[:,0] ) and np.all( x <= bounds[:,1] )]
			X, y = X[I], y[I]
		return bounds, X, y

	def _update_unknown_dims (self, bounds, **kwargs):
		new_bounds, X, y = self._update(bounds.copy(), **kwargs)
		for d in self.known_dim:
			new_bounds[d] = bounds[d]
		return new_bounds, X, y


class EmpiricalBoxBoundUpdate (BoundUpdate):
	"""
	Return smallest box bounding all observed feasible points
	"""
	def __init__ (self, *args, stretch=0.5, **kwargs):
		super().__init__(*args, **kwargs)
		assert isinstance(stretch, (int,float)) and stretch >= 0
		self.stretch = stretch

	def _update (self, bounds, **kwargs):
		if ('X' in kwargs and 'Y' in kwargs):
			X, y  = kwargs.get('X'), kwargs.get('Y')
		elif not 'model' in kwargs:
			raise ValueError('Need data to update empirical box bounds')
		else:
			model = kwargs.get('model')
			X, y  = model.X.copy(), model.Y[:,0].copy()
		if y.ndim > 1:
			y = y.flatten()
		assert X.shape[0] == len(y), 'Data not of equal length!'

		# Find feasible observations
		I = np.arange(len(y))[y>=0]
		if len(I) > 2*len(bounds):
			# Stretched empirical box
			Xf = X[I]
			for d in range( bounds.shape[0] ):
				if d in self.known_dim: continue
				bounds[d] = np.array([ np.min(Xf[:,d]), np.max(Xf[:,d]) ])
				bounds[d] = stretch_bounds(bounds[d], self.stretch)
		return bounds, X, y


class MaximumProbableBoundUpdate (BoundUpdate):
	r"""
	Return largest possible bounding box subject to
		\mu(x) + 2*\sigma(x) >= 0
	"""
	def __init__ (self, *args, stretch=0.5, average=True, update_iters=[], **kwargs):
		super().__init__(*args, **kwargs)
		assert isinstance(stretch, (int,float)) and stretch >= 0
		self.stretch = stretch
		self.average = average
		self.update_iters = [ int(i) for i in update_iters ]

	def _update (self, bounds, **kwargs):
		if not 'model' in kwargs:
			raise ValueError('Need feasibility model to update bounds')
		model = kwargs.get('model')
		i = kwargs.get('i')
		y = model.Y[:,0]
		
		new_bounds = bounds.copy()
		num_feas   = np.sum(np.ones(len(y))[y>=0]) > 2*len(bounds)
		iter_match = len( self.update_iters ) == 0 or i+1 in self.update_iters
		
		if num_feas and iter_match:
			X_init = [ x for x in model.X if self._constraint(model, x) >= 0 ]
			if len(X_init) < 2:
				X_init = model.X.copy()

			for j in range( model.X.shape[1] ):
				if j in self.known_dim: continue

				## Lower bound
				lobnd = lambda x: self._objective_j(x, j)
				xlow  = np.inf
				for x0 in X_init:
					res = self._slsqp_min(model, lobnd, x0, bounds, j, low=True)
					if res['success']:
						xlow = np.min(( xlow, res['x'][j] ))
				if xlow < np.inf:
					new_bounds[j,0] = xlow

				## Upper bound
				upbnd = lambda x: self._objective_j(x,j,negate=True)
				xhigh = -np.inf
				for x0 in X_init:
					res = self._slsqp_min(model, upbnd, x0, bounds, j, low=False)
					if res['success']:
						xhigh = np.max(( xhigh, res['x'][j] ))
				if xhigh > -np.inf:
					new_bounds[j,1] = xhigh
			new_bounds = stretch_bounds(new_bounds, self.stretch)
			
		if self.average:
			new_bounds = 0.5 * (bounds + new_bounds)
		return new_bounds, model.X, model.Y[:,0]

	def _objective_j (self, x, j, negate=False):
		f     = -x[j] if negate else x[j]
		df    = np.zeros(x.shape)
		df[j] = -1. if negate else 1.
		return f, df

	def _tmp_bounds (self, model, bounds, j, low=True):
		"""
		Our x is not allowed to be 'optimised' such that the new bound would
		not include points that we have evaluated and know are feasible
		"""
		Xtmp = model.X[model.Y[:,0]>=0]
		bnds = bounds.copy()
		if low:
			bnds[j,1] = np.max((bnds[j,1], np.min(Xtmp[:,j])))
		else:
			bnds[j,0] = np.min((bnds[j,0], np.max(Xtmp[:,j])))
		return bnds.tolist()

	def _constraint (self, model, x, factor=1., grad=False):
		r"""
		Constraint: \mu(x) + 2*\sigma(x) >= 0
		"""   
		# Constraint
		m, s = model.predict_noiseless(x[None,:])
		m, s = m[0], np.sqrt(s[0])
		c    = m + 2*s
		if not grad:
			return c * factor

		# Gradient
		dm, ds = model.predictive_gradients(x[None,:])
		dm     = dm.reshape((1, len(x)))
		ds     = ds / (2. * s)
		dc     = dm + 2*ds

		return dc * factor

	def _slsqp_min (self, model, func, x0, bounds, j, low=True):
		# Objective and gradients
		f  = lambda x: func(x)[0]
		df = lambda x: func(x)[1]

		# This mult. factor helps with "no positive search direction" issues
		factor = 1.
		for n in range( 4 ):
			# Constraint
			cons = {'type': 'ineq',
			    'fun' : lambda x: self._constraint(model, x, factor),
			    'jac' : lambda x: self._constraint(model, x, factor, True)}

			# Optimise
			bnds = self._tmp_bounds(model, bounds, j, low=low)
			dic  = {'method':'SLSQP', 'constraints':(cons,), 'bounds':bnds}
			res  = minimize(f, x0, jac=df, **dic)
			if res['status'] == 8:
				factor *= 100.
			else:
				return res
		return res