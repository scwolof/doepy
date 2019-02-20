
"""
The slsqp code is taken from scipy.optimize.slsqp._minimize_slsqp
"""

import numpy as np 
from scipy.optimize._slsqp import slsqp as scipy_slsqp

def slsqp (problem_instance, x0, maxiter=100, ftol=1.0E-6):
	"""
	Wrapping function for scipy SLSQP function to solve control problems with
	inequality constraints.
	"""
	# Transform x0 into an array.
	x = np.asfarray(x0).flatten()

	# m = The total number of constraints
	m = int(problem_instance.num_constraints)
	# la = The number of constraints, or 1 if there are no constraints
	la = max([1, m])
	# n = The number of independent variables
	n = len(x)

	# Define the workspaces for SLSQP
	n1     = n + 1
	mineq  = m + n1 + n1
	len_jw = mineq
	len_w  = (3*n1+m)*(n1+1)+(n1+1)*(mineq+2) + 2*mineq+(n1+mineq)*n1 \
	          + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
	w  = np.zeros(len_w)
	jw = np.zeros(len_jw)

	# Decompose bounds into xl and xu
	bnds = np.array(problem_instance.bounds, float)
	if bnds.shape[0] != n:
		raise IndexError('SLSQP Error: the length of bounds is not '
						 'compatible with that of x0.')

	with np.errstate(invalid='ignore'):
		bnderr = bnds[:, 0] > bnds[:, 1]
	if bnderr.any():
		raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
						 ', '.join(str(b) for b in bnderr))
	xl, xu = bnds[:, 0], bnds[:, 1]

	# Mark infinite bounds with nans; the Fortran code understands this
	infbnd = ~np.isfinite(bnds)
	xl[infbnd[:, 0]] = np.nan
	xu[infbnd[:, 1]] = np.nan

	# Clip initial guess to bounds 
	# (SLSQP may fail with bounds-infeasible initial point)
	have_bound    = np.isfinite(xl)
	x[have_bound] = np.clip(x[have_bound], xl[have_bound], np.inf)
	have_bound    = np.isfinite(xu)
	x[have_bound] = np.clip(x[have_bound], -np.inf, xu[have_bound])

	# Initialize the iteration counter and the mode value
	mode    = np.array(0, int)
	acc     = np.array(ftol, float)
	majiter = np.array(maxiter, int)

	# Initialize internal SLSQP state variables
	float_zero = lambda n: [ np.array(0, float) ] * n
	alpha, f0, gs, h1, h2, h3, h4, t, t0, tol = [ np.array(0, float) ] * 10
	iexact, incons, ireset, itermx, line, n1, n2, n3 = [ np.array(0, int) ] * 8

	while 1:

		if mode == 0 or mode == 1: 
			"""
			Update objective function and constraints, as well as their 
			corresponding gradients. (It is assumed that gradients are 
			computed as part of objective and constraint evaluation.)
			"""
			f, c, df, dc = func(x)

			f  = float(np.asarray(f))
			df = np.append(df, 0.0)
			dc = np.c_[ dc, np.zeros( la ) ]
			
		# Call SLSQP
		scipy_slsqp(m, 0, x, xl, xu, f, c, df, dc, acc, majiter, mode, w, jw,
		            alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
		            iexact, incons, ireset, itermx, line, n1, n2, n3)

		# If exit mode is not -1 or 1, slsqp has completed
		if abs(mode) != 1:
			break

	status  = int(mode)
	sucess  = status == 0
	message = {-1: "Gradient evaluation required (df & dc)",
	            0: "Optimization terminated successfully.",
	            1: "Function evaluation required (f & c)",
	            2: "More equality constraints than independent variables",
	            3: "More than 3*n iterations in LSQ subproblem",
	            4: "Inequality constraints incompatible",
	            5: "Singular matrix E in LSQ subproblem",
	            6: "Singular matrix C in LSQ subproblem",
	            7: "Rank-deficient equality constraint subproblem HFTI",
	            8: "Positive directional derivative for linesearch",
	            9: "Iteration limit exceeded"}[ status ]

	return {'x':x, 'f':f, 'success':success, 'status':status, 'message':message}
