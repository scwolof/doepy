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
from scipy.optimize import fmin_l_bfgs_b

def multistart_fmin_l_bfgs_b (f, X_init, bounds, **kwargs):
    """
    Run fmin_l_bfgs_b initialised at points X_init.
    Return best result.
    """
    xbest = np.ones(X_init.shape[1]) * np.nan
    ybest = np.inf
    if not isinstance(bounds, list):
        try: bounds = bounds.tolist()
        except: raise ValueError('bounds must be a list')
    for x in X_init:
        xopt,fopt,d = fmin_l_bfgs_b(f, x, bounds=bounds, **kwargs)
        if fopt < ybest:
            ybest = fopt
            xbest = xopt
    return xbest, ybest

def multistart_points (f, bounds, N_test=1000, N_return=10, args=()):
    """
    Generate points to initialise optimisation from.
    Samples N_test points and returns N_return most promising candidates.
    """
    X = np.vstack([ b[0] + (b[1]-b[0])*np.random.rand(N_test) for b in bounds ]).T
    I = np.argsort( f(X, *args) )[:N_return]
    return X[I]