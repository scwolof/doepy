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

from .find_active_dims import find_active_dims

def generate_training_data (f, x_bounds, u_bounds, p_bounds=None, active_dims=None,
                   num_data_points_per_num_dim=None):
    """
    Generate training data sets for each target dimension of f
    """
    bounds = (x_bounds, u_bounds)
    if active_dims is None:
        ad_bounds   = bounds if p_bounds is None else bounds + (p_bounds,)
        active_dims = find_active_dims(f, ad_bounds)

    # Generate meshgrids of x and u
    X, U = generate_locations(bounds, active_dims, num_data_points_per_num_dim)
    # Generate observations at points [x,u,(p)] in (X,U)
    Y    = generate_observations(f, X, U, active_dims, p_bounds=p_bounds)
    if p_bounds is None:
        return X, U, Y
    return X, U, Y[0], Y[1]


def generate_locations (bounds, active_dims, num_data_points_per_num_dim=None):
    """
    Generate training locations

    bounds   (numpy.ndarray, or list/tuple of numpy.ndarray)
             If bounds = (bounds_1, ..., bounds_D), return grids X_1, ..., X_D, 
             which are lists with training data for each target dimension

    active_dims   List of lists with "active" input dimensions for each target
                  dimension (see find_active_dims)

    num_data_points_per_num_dim    How many points to sample along each axis for
                                   target dimensions with n active input dims.
    """
    if isinstance(bounds, (list,tuple)):
        Dss    = np.cumsum([0] + [b.shape[0] for b in bounds])
        bounds = np.vstack(bounds)
    else:
        Dss = [0, bounds.shape[0]]
    Ns = num_data_points_per_num_dim
    if Ns is None:
        Ns = [101, 21, 10, 6, 4, 3]

    # Number of target dimensions
    E = len( active_dims )
    # Number of sets to return (e.g. [X, U] -> 2)
    N = len(Dss) - 1
    T = [ [None]*E for _ in range(N) ]

    for i in range( E ):
        # Generate meshgrid of x and u
        ad = active_dims[i]
        ns = np.sum([1 for i in ad if i < Dss[-1]])
        ns = Ns[-1] if ns > len(Ns) else Ns[ns-1]
        t  = []
        for j in range( Dss[-1] ):
            if j in ad:
                t.append( np.linspace(*bounds[j],ns) )
            else:
                t.append( [bounds[j,0]] )
        t = np.vstack([ ti.flatten() for ti in np.meshgrid(*t) ]).T
        
        for n in range(N):
            T[n][i] = t[:,Dss[n]:Dss[n+1]]
    return T


def generate_observations (f, X, U, active_dims, p_bounds=None):
    """
    Combine (if possible) the training data sets X and U (where the active_dims)
    do not interfere, draw random parameter values P if p_bounds is not None, 
    and evaluate Y = f(X,U,(P)). Return (P),Y
    """
    E = len(active_dims)
    assert len(X) == E
    assert len(U) == E
    
    Dx = X[0].shape[1]
    Du = U[0].shape[1]
    Ns = [ x.shape[0] for x in X ]
    
    combo = [i for i in range(len(X))] #find_smallest_combo( active_dims, Ns )
    
    Yt = [ None ] * E
    Pt = [ None ] * E
    for e in range( E ):
        if not e in combo:
            break
        
        nx = np.max([ x.shape[0] for i,x in enumerate(X) if combo[i] == e ])
        Xt = np.ones(( nx, Dx )) * X[0][0]  # Feasible X initialisation
        Ut = np.ones(( nx, Du )) * U[0][0]  # Feasible U initialisation
        
        # Combine locations in combo
        for i in range( E ):
            if not combo[i] == e:
                continue
            for a in active_dims[i]:
                if a < Dx:
                    x = X[i][:,a]
                    Xt[:len(x),a] = x
                elif Dx <= a < Dx+Du:
                    au = a - Dx
                    u  = U[i][:,au]
                    Ut[:len(u),au] = u
                    
        # Evaluate function for combined training sets
        if p_bounds is None:
            Y = np.array([ f(x,u) for x,u in zip(Xt, Ut) ])
        else:
            # Sample parameter values
            b1, b2 = p_bounds[:,0], p_bounds[:,1]
            P = np.random.uniform( b1, b2, (Xt.shape[0],b1.shape[0]) )
            Y = np.array([ f(x,u,p) for x,u,p in zip(Xt, Ut, P) ])
            
        for i in range( E ):
            if not combo[i] == e:
                continue
            Ni    = X[i].shape[0]
            Yt[i] = Y[:Ni,i]
            if p_bounds is not None:
                Pt[i] = P[:Ni]
    return Yt if p_bounds is None else (Pt, Yt)


def combo_size ( combo, Ns ):
    E = len( combo )
    W = np.zeros( E, int )
    for c,n in zip( combo, Ns ):
        W[c] = np.max(( n, W[c] ))
    return np.sum(W)

def is_feasible_combo ( l, active_dims ):
    E = len(active_dims)
    assert len(l) == E
    if E == 1:
        return True
    for i in range(1,E):
        for j in range(i):
            if l[i] == l[j]:
                ai = active_dims[i]
                aj = active_dims[j]
                same_length = len(ai) == len(aj)
                share_dim   = np.any([ a in aj for a in ai ])
                if (not same_length) and share_dim:
                    return False
    return True

def find_unique_combos (num_sets):
    assert isinstance(num_sets, int) and num_sets >= 1
    lists = [[0]]
    for nl in range(num_sets-1):
        _lists = []
        for l in lists:
            _lists += [ l.copy()+[m] for m in range(np.max(l)+2) ]
        lists = _lists.copy()
    return lists

def find_feasible_combos ( active_dims ):
    E     = len( active_dims )
    lists = find_unique_combos( E )
    return [ l for l in lists if is_feasible_combo(l, active_dims) ]

def find_smallest_combo ( active_dims, Ns ):
    lists = find_feasible_combos( active_dims )
    sizes = [ combo_size(combo, Ns) for combo in lists ]
    return lists[ np.argmin(sizes) ]