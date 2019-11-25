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
from scipy.special import erf, erfinv

class StateConstraint:
    def __init__ (self, bounds, **kwargs):
        self.bounds = np.asarray( bounds )

        if self.bounds.ndim == 2:
            # Constant bounds
            assert self.bounds.shape[1] == 2
            self.num_states = self.bounds.shape[0]
        elif self.bounds.ndim == 3:
            # Time-dependent bounds
            assert self.bounds.shape[2] == 2
            self.num_steps  = self.bounds.shape[0]
            self.num_states = self.bounds.shape[1]
        
        if 'conf' in kwargs:
            # Constant bounds
            self.conf = kwargs['conf']
            
    @property
    def has_added_variables (self):
        return False

    def num_constraints (self):
        # Number of individual constraints constructed by class
        raise NotImplementedError 
        
    def __call__ (self, M, S, step=None, grad=False):
        """
        Input:
        M   [ num_meas ]              Matrix of predictive means
        S   [ num_meas x num_meas ]   Matrix of predictive covariances
        grad                          Return gradients wrt M and S

        Output:
        c      Constraint score for M, S
        dcdM   Gradient d c / d M
        dcdS   Gradient d c / d S
        """
        raise NotImplementedError
        

class ConstantMeanStateConstraint (StateConstraint):
    r"""
    Mean constraint:
        bounds[i,0] <= \mu_i(t) <= bounds[i,1]
    """
    def __init__ (self, bounds):
        super().__init__ (bounds)
        assert self.bounds.ndim == 2

    def num_constraints (self):
        return 2 * self.num_states
        
    def __call__ (self, M, S, step=None, grad=False):
        C = np.zeros( 2 * self.num_states )
        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )
        
        for i in range(self.num_states):
            C[2*i]   = M[i] - self.bounds[i,0]
            C[2*i+1] = self.bounds[i,1] - M[i]
            if grad:
                dCdM[2*i,  i] = 1.
                dCdM[2*i+1,i] = -1.
        return C if not grad else (C, dCdM, dCdS)
        

class MovingMeanStateConstraint (StateConstraint):
    r"""
    Mean constraint:
        bounds[t,i,0] <= \mu_i(t) <= bounds[t,i,1]
    """
    def __init__ (self, bounds):
        super().__init__ (bounds)
        assert self.bounds.ndim == 3

    def num_constraints (self):
        return 2 * self.num_states
        
    def __call__ (self, M, S, step, grad=False):
        step = np.min(( self.num_steps-1, step ))
        C    = np.zeros( 2 * self.num_states )
        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )
        
        for i in range(self.num_states):
            C[2*i]   = M[i] - self.bounds[step,i,0]
            C[2*i+1] = self.bounds[step,i,1] - M[i]
            if grad:
                dCdM[2*i,  i] = 1.
                dCdM[2*i+1,i] = -1.
        return C if not grad else (C, dCdM, dCdS)


class SingleChanceStateConstraint (StateConstraint):
    r"""    
    Chance constraint:
        conf = confidence
        r = distance from the mean value for a defined tolerance
        P( \mu_i(t) + alpha * S <= bounds[i,1] ) > conf
        P( \mu_i(t) - alpha * S >= bounds[i,1] ) > conf
    """   
    def __init__ (self, bounds, conf=0.99):
        super().__init__ (bounds)
        assert self.bounds.ndim == 2
        self.conf  = conf
        self.alpha = np.sqrt(2) * erfinv(self.conf)

    def num_constraints (self):
        return 2 * self.num_states

    def __call__ (self, M, S, step=None, grad=False):
        C = np.zeros( 2 * self.num_states )
        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )

        for i in range(self.num_states):
            C[2*i]   = M[i] - self.alpha*np.sqrt(S[i,i]) - self.bounds[i,0];
            C[2*i+1] = self.bounds[i,1]- self.alpha*np.sqrt(S[i,i]) - M[i];

            if grad:
                dCdM[2*i,  i]   = 1.
                dCdM[2*i+1,i]   = -1.
                dCdS[2*i,  i] = -self.alpha/(2*np.sqrt(S[i,i]))
                dCdS[2*i+1,i] = -self.alpha/(2*np.sqrt(S[i,i]))
        return C if not grad else (C, dCdM, dCdS)


class JointTimeChanceStateConstraint (StateConstraint):
    r"""    
    Chance constraint:
        P1(\mu_i(t)+r*S <= bounds[i,1]) > 1-eps
        P2(\mu_i(t)-r*S >= bounds[i,1]) > 1-eps
    """   
    def __init__ (self, bounds, conf=0.95):
        super().__init__ (bounds,**kwargs)
        assert self.bounds.ndim == 2

    def num_constraints (self):
        return 2 * self.num_states

    def __call__ (self, M, S, step=None, grad=False):
        C = np.zeros( 2 * self.num_states )
        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )

        for i in range(self.num_states):
            div = 1. / np.sqrt(S[i,i])
            r1  = (M[i] - self.bounds[i,0]) * div
            r2  = (self.bounds[i,1] - M[i]) * div
            P1  = 0.5 * ( 1 - erf(r1 / np.sqrt(2)) )
            P2  = 0.5 * ( 1 - erf(r2 / np.sqrt(2)) )

            C[2*i]   = P1
            C[2*i+1] = P2

            if grad:
                diff  = M[i] - self.bounds[i]
                diff2 = np.exp( -0.5 * diff**2 / S[i,i] )
                div1  = 1. / np.sqrt(2*np.pi*S[i,i])
                div2  = 1. / ( np.sqrt(2*np.pi)*S[i,i] )
                dCdM[2*i,  i]   = -diff2[0] * div1
                dCdM[2*i+1,i]   = -diff2[1] * div1
                dCdS[2*i,  i,i] =  diff[0] * diff2[0] * div2
                dCdS[2*i+1,i,i] = -diff[1] * diff2[1] * div2
        return C if not grad else (C, dCdM, dCdS)