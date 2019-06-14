
"""
The slsqp code is taken from scipy.optimize.slsqp._minimize_slsqp
"""

import numpy as np 
from scipy.optimize._slsqp import slsqp as scipy_slsqp

def slsqp (problem_instance, u0, maxiter=30, ftol=1.0E-5, log_callback=None,
           maxinneriter=10, stepdesc=0.1, qptol=1e-6, debug=False):
    """
    Wrapping function for scipy SLSQP function to solve control problems with
    inequality constraints.

    u0:         optimisation starting point
    maxiter:    max number of iterations of the SQP solver
    stepdesc:   step of QP solver
    qptol:      tolerance used for the qp solver (used for constraint tolerance)
    debug:      print iterations calculations
    """
    # Transform u0 into an array.
    u = np.asfarray(u0).flatten()

    # Make sure log_callback is callable function
    if log_callback is not None:
        assert callable(log_callback), 'log_callback must be callable function'

    # m = The total number of constraints
    m = int(problem_instance.num_constraints)
    # la = The number of constraints, or 1 if there are no constraints
    la = max([1, m])
    # n = The number of independent variables
    n = len(u)

    # Define the workspaces for SLSQP
    n1     = n + 1
    mineq  = m + n1 + n1
    len_jw = mineq
    len_w  = (3*n1+m)*(n1+1)+(n1+1)*(mineq+2) + 2*mineq+(n1+mineq)*n1 \
              + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    w  = np.zeros(len_w)
    jw = np.zeros(len_jw)

    # Decompose bounds into ul and uu
    bnds = np.array(problem_instance.bounds, float)
    if bnds.shape[0] != n:
        raise IndexError('SLSQP Error: the length of bounds is not '
                         'compatible with that of u0.')

    with np.errstate(invalid='ignore'):
        bnderr = bnds[:, 0] > bnds[:, 1]
    if bnderr.any():
        raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                         ', '.join(str(b) for b in bnderr))
    ul, uu = bnds[:, 0], bnds[:, 1]

    # Mark infinite bounds with nans; the Fortran code understands this
    infbnd = ~np.isfinite(bnds)
    ul[infbnd[:, 0]] = np.nan
    uu[infbnd[:, 1]] = np.nan

    # Clip initial guess to bounds 
    # (SLSQP may fail with bounds-infeasible initial point)
    have_bound    = np.isfinite(ul)
    u[have_bound] = np.clip(u[have_bound], ul[have_bound], np.inf)
    have_bound    = np.isfinite(uu)
    u[have_bound] = np.clip(u[have_bound], -np.inf, uu[have_bound])

    # Initialize the iteration counter and the mode value
    mode    = np.array(0, int)
    acc     = np.array(ftol, float)
    majiter = np.array(maxiter, int)

    # Initialize internal SLSQP state variables
    f0, gs, h1, h2, h3, h4, t, t0 = [ np.array(0, float) ] * 8
    #iexact, incons, ireset, line, itermx, n1, n2, n3 = [ np.array(0, int) ] * 8
    iexact, incons, ireset, line, n1, n2, n3 = [ np.array(0, int) ] * 7
    alpha  = np.array(stepdesc, float)
    tol    = np.array(qptol, float)
    itermx = np.array(maxinneriter, int)

    # Rescaling factor to try to recover from mode 8 "Positive directional 
    # derivative for linesearch" when running SLSQP optimisation algorithm
    saved_f        = []
    max_saved_f    = 20
    rescale_factor = None

    if debug:
        print("%5s %16s %16s %16s %16s %16s"\
            %("NIT","OBJFUN","FNORM","CMIN","CNORM","F+STEP"))

    while 1:
        if mode == 0 or mode == 1: 
            """
            Update objective function and constraints, as well as their 
            corresponding gradients. (It is assumed that gradients are 
            computed as part of objective and constraint evaluation.)
            """
            f, c, df, dc = problem_instance(u)

            # Initial rescaling factor
            if rescale_factor is None:
                fact = np.abs(f)
                if fact > 5:
                    rescale_factor = 2. / fact
                else:
                    rescale_factor = 1.
                #print('Initial rescaling: %f'%(rescale_factor))

            # Write to log
            if log_callback is not None:
                log_callback(u, f, c, df, dc, rescale_factor)

            # Keep record of latest function evaluations
            # - for computing rescaling factor
            if len(saved_f) < max_saved_f:
                saved_f.append(f)
            else:
                saved_f = saved_f[1:] + [f]

            f  = float(np.asarray(f)) * rescale_factor
            df = np.append(df, 0.0) * rescale_factor
            dc = np.c_[ dc, np.zeros( la ) ]

            if np.any([np.any(np.isnan(mat)) for mat in [u, f, c, df, dc]]):
                mode = 10
                break

        # Call SLSQP
        scipy_slsqp(m, 0, u, ul, uu, f, c, df, dc, acc, majiter, mode, w, jw,
                    alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
                    iexact, incons, ireset, itermx, line, n1, n2, n3)

        # Enforce maximum number of iterations
        itermx = np.array(maxiter, int)

        if debug and mode == 1:
            fnorm, cnorm = np.linalg.norm(df), np.linalg.norm(dc)
            print("%5i %16.6E %16.6E %16.6E %16.6E %16.6E"\
                %(majiter, f, fnorm, np.min(c), cnorm, f+0.01*np.sum(df)))

        # If exit mode is not -1 or 1, slsqp has typically completed
        # HOWEVER: we do not necessarily want to terminate on mode 8
        if abs(mode) != 1:
            if mode == 8:
                alpha = 0.5 * (1. + alpha)
                """
                abs_non_zero_f = [np.abs(sf) for sf in saved_f if sf != 0]
                if len( abs_non_zero_f ) == 0:
                    break
                rescale_factor = 2. / np.mean(abs_non_zero_f)
                f  *= rescale_factor
                df *= rescale_factor
                """
                #print('Changing rescaling: %f'%(rescale_factor))
            else:
                break

    u   = u.reshape( u0.shape )
    f   = f / rescale_factor
    df  = df[:-1] / rescale_factor
    dc  = dc[:,:-1]
    res = {'u':u, 'f':f, 'c':c, 'df':df, 'dc':dc}

    status  = int(mode)
    success = status == 0
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
                9: "Iteration limit exceeded",
               10: "NaN detected"}[ status ]
    res.update( {'success':success, 'status':status, 'message':message} )

    return res
