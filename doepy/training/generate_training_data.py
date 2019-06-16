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
                   num_data_points_per_num_dim=None, return_active_dims=False):
    """
    Generate training data sets for each target dimension of f
    """
    has_p = p_bounds is not None

    bounds = (x_bounds, u_bounds)
    if active_dims is None:
        ad_bounds   = bounds if not has_p else bounds+(p_bounds,)
        active_dims = find_active_dims(f, ad_bounds)

    # Generate meshgrids of x and u
    X, U = generate_locations(bounds, active_dims, num_data_points_per_num_dim)
    # Generate observations at points [x,u,(p)] in (X,U)
    Y = generate_observations(f, X, U, active_dims, p_bounds=p_bounds)
    if has_p:
        P, Y = Y

    # Check for NaN/inf values
    N = len( active_dims )
    for n in range(N): # For all target dimensions
        z = Y[n]
        D = [ i for i in range(len(z)) if np.isnan(z[i]) or np.isinf(z[i]) ]
        if len(D) > 0:
            # Delete invalid training data
            X[n] = np.delete(X[n], D, axis=0)
            U[n] = np.delete(U[n], D, axis=0)
            Y[n] = np.delete(Y[n], D, axis=0)
            if has_p:
                P[n] = np.delete(P[n], D, axis=0)

    # Concatenate training data
    T  = (X, U, Y) if not has_p else (X, U, P, Y)

    if return_active_dims:
        return T, active_dims
    return T


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




class TrainingDimensionsCombo:
    def __init__ (self, combo):
        # Combination of training target dimensions
        self.y = combo
        self.E = len(self.y)
        # Combination of training input dimensions
        self.x = None
        # Number of required function evalutations
        self.size = None
        
    def is_feasible (self, active_dims):
        """
        Given active_dims, is it possible to train the
        target dimensions together according to self.y?
        -> If True, will assign input combo to self.x
        """
        assert len( active_dims ) == self.E
        if self.x is not None:
            return True
        
        # Do we have more than one target dimension?
        if self.E == 1:
            self.x = [active_dims]
            return True
        
        # Are active_dims obviously incompatible?
        for i in range(1, self.E):
            for j in range(i):
                if self.y[i] == self.y[j]:
                    ai = active_dims[i]
                    aj = active_dims[j]
                    same_length = len(ai) == len(aj)
                    share_dim   = np.any([ a in aj for a in ai ])
                    if (not same_length) and share_dim:
                        return False
        
        # Can we construct feasible combination of input dimensions?
        combo = []
        for i in range( self.E ):
            dims = [ active_dims[j] for j,c in enumerate(self.y) if c==i ]
            if len(dims) > 0:
                is_feasible, graph = self.is_feasible_combo( dims )
                if not is_feasible:
                    return False
                combo.append(graph)
                
        self.x = combo
        return True
    
    def is_feasible_combo ( self, dims ):
        """
        Try to construct self.x
        """
        D = [ len(d) for d in dims ]
        if np.all( np.asarray(D)==D[0] ):
            # All active_dims are of the same length
            B, L = self.is_feasible_combo_equal_length(dims)
            return B, [L]
        else:
            # There are active_dims in disjunct sets
            return self.is_feasible_combo_disjunct(dims)
                    
    def is_feasible_combo_equal_length ( self, dims ):
        """
        All chosen active_dims in dims are of equal length
        -> See if they can be combined
        e.g. [ [1,2,3], [1,3,4], [1,4,5] ] -> [ 1, [2,4], [3,5] ]
        """
        N = len(dims)
        D = len(dims[0])

        # Find all indices
        indices = []
        for l in dims:
            for d in l:
                if not d in indices:
                    indices.append(d)
        indices.sort()

        # Find all mutual indices
        mutual = indices.copy()
        remain = []
        for d in indices:
            for l in dims:
                if not d in l:
                    mutual.remove(d)
                    if not d in remain:
                        remain.append(d)
                    break

        if len(mutual) == D:
            return True, mutual

        remain.sort()
        free_ind = np.arange(len(mutual), D).tolist()
        Nf    = len(free_ind)
        combo = mutual.copy()

        combos = self.find_unique_combos(len(remain), Nf)
        if Nf > 1:
            combos = combos[1:]
        for C in combos:
            okay = True
            inds = [ [remain[i] for i,c in enumerate(C) if c==j] for j in range(Nf) ]
            for ind in inds:
                if not self.includes_only_one( dims, ind ):
                    okay = False
                    break
            if okay:
                return True, combo + inds

        return False, None
    
    def is_feasible_combo_disjunct ( self, dims ):
        """
        There are disjunct sets that we might be able to combine
        e.g. [ [0,1], [2,3], [3,4] ] ->  [ [], [3, []] ]
        """
        # active_dims lengths
        lengths = []
        for d in dims:
            leng = len(d)
            if not leng in lengths:
                lengths.append(leng)
        lengths.sort()
        
        A = []
        for l in lengths:
            s = []
            for d in dims:
                if len(d) == l:
                    s.append(d)
            a = self.is_feasible_combo_equal_length(s)
            if not a[0]:
                return False, None
            A.append(a[1])
            
        return True, A

    def find_unique_combos ( self, num_sets, max_ind ):
        """
        Find all unique, ordered combinations of input dimensions
        """
        lists = [[0]]
        for nl in range(num_sets-1):
            _lists = []
            for l in lists:
                mind    = np.min((np.max(l)+2, max_ind))
                _lists += [ l.copy()+[m] for m in range(int(mind)) ]
            lists = _lists.copy()
        return lists

    def includes_only_one ( self, dims, inds ):
        """
        Make sure that each list in dims contains at
        most one of the indices in the the inds list
        """
        for l in dims:
            has_one = False
            for i in inds:
                if i in l:
                    if has_one:
                        return False
                    has_one = True
        return True
        
    def get_num_per_output (self, dim, Ns ):
        n  = Ns[-1] if dim > len(Ns) else Ns[dim-1]
        return n, n**dim
    
    def get_num_required_func_evals ( self, Ns ):
        assert self.x is not None
        func  = self._get_num_required_func_evals
        return np.sum([ func(Ns, x) for x in self.x ])
    
    def _get_num_required_func_evals ( self, Ns, X ):
        return np.max([ self.get_num_per_output(len(x),Ns)[1] for x in X ])
            


class TrainingDataGenerator:
    def __init__ (self, active_dims, bounds,\
                  num_data_points_per_num_dim=[101, 26, 10, 6, 4, 3]):
        self.active_dims    = active_dims
        self.bounds         = bounds
        self.num_per_output = num_data_points_per_num_dim
        self.num_dim        = int( 1 + np.max([ np.max(d) for d in self.active_dims ]) )
        self._combo         = None
        
    def get_num_per_output (self, dim ):
        Ns = self.num_per_output
        n  = Ns[-1] if dim > len(Ns) else Ns[dim-1]
        return n, n**dim

    def find_smallest_combo ( self ):
        """
        Find combination of input dimensions that will result in the fewest
        number of model evaluations to generate the training data set.
        """
        if self._combo is None:
            # Find feasible combinations
            lists = self.find_feasible_combos()
            # Find smallest feasible combination
            Ns  = self.num_per_output
            ind = np.argmin([ l.get_num_required_func_evals(Ns) for l in lists])
            # Combination of training dimensions
            self._combo = lists[ ind ]
        return self._combo

    def find_feasible_combos ( self ):
        """
        Find feasible combinations of input dimensions
        """
        combos = self.find_unique_combos()
        return [ c for c in combos if c.is_feasible( self.active_dims ) ]

    def find_unique_combos ( self ):
        """
        Find all unique, ordered combinations of input dimensions
        """
        num_sets = len( self.active_dims )
            
        lists = [[0]]
        for nl in range(num_sets-1):
            _lists = []
            for l in lists:
                # [0]   -> [0,0], [0,1]
                # [0,1] -> [0,1,0], [0,1,1], [0,1,2]
                # etc
                _lists += [ l.copy()+[m] for m in range( np.max(l)+2 ) ]
            lists = _lists.copy()
        lists = [ TrainingDimensionsCombo(l) for l in lists ]
        return lists
    
    def grid_locations( self ):
        combo = self.find_smallest_combo()
        # Number of output sets
        nout = len( combo.x )
        # Number of input dimensions
        nin  = self.bounds.shape[0]
        # Number of training data points per dimension
        Ns  = self.num_per_output
        
        X = []
        for x in combo.x:
            nx = combo._get_num_required_func_evals(Ns, x)
            Xs = np.ones((nx,1)) * np.mean(self.bounds,axis=1)[None,:]
            for dims in x:
                xs = self._grid_locations_equal_dims(dims)
                for c1 in dims:
                    if isinstance(c1, int):
                        c1 = [c1]
                    for c2 in c1:
                        Xs[:xs.shape[0],c1] = xs[:,c1]
            X.append(Xs)
        return X
        
    def _grid_locations_equal_dims (self, dims):
        L  = len(dims)
        n  = self.get_num_per_output(L)[0]
        Z  = np.meshgrid(*[np.arange(n)]*L)
        Z  = [ z.flatten() for z in Z ]
        N  = len( Z[0] )
        #Xs = np.mean(self.bounds,axis=1)[None,:] * np.ones((N,1))
        Xs = np.zeros(( N, len(self.bounds) ))
        for c1,z in zip(dims,Z):
            if isinstance(c1,int):
                c1 = [c1]
            for c2 in c1:
                zz = np.linspace(*self.bounds[c2], n)
                Xs[:,c2] = np.array([zz[z1] for z1 in z])
        return Xs
    
    def generate_observations (self, f, X=None):
        if X is None:
            X = self.grid_locations()
        combo = self.find_smallest_combo()
        # Need to separate into x, u (and add parameter values)
        Y = [ np.array([ f(x) for x in Xs ]) for Xs in X ]
        X = [ X[i] for i in combo.y ]
        Y = [ Y[i][:,[j]] for j,i in enumerate(combo.y) ]
        return X, Y