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
        #print( 'mean' )
        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )
        
        for i in range(self.num_states):
            C[2*i]   = M[i] - self.bounds[i,0]
            #print('lower: %d'%i)
            #print(self.bounds[i,0])
            C[2*i+1] = self.bounds[i,1] - M[i]
            #print('upper: %d'%i)
            #print(self.bounds[i,1])
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
    def __init__ (self, bounds, conf=0.95):
        super().__init__ (bounds)
        assert self.bounds.ndim == 2
        self.conf  = conf
        self.r = np.sqrt(2) * erfinv(self.conf)
        

    def num_constraints (self):
        return 2 * self.num_states
    
    def __call__ (self, M, S, step=None, grad=False):
        C = np.zeros( 2 * self.num_states )
        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )
        
        for i in range(self.num_states):
            C[2*i]   = M[i] - self.r*np.sqrt(S[i,i]) - self.bounds[i,0];
            C[2*i+1] = self.bounds[i,1]- self.r*np.sqrt(S[i,i]) - M[i];
            
            if grad:
                dCdM[2*i,  i] = 1.
                dCdM[2*i+1,i] = -1.
                dCdS[2*i,  i, i] = -self.r/(2*np.sqrt(S[i,i]))
                dCdS[2*i+1,i, i] = -self.r/(2*np.sqrt(S[i,i]))
        return C if not grad else (C, dCdM, dCdS)
    
class PointwiseChanceStateConstraint (StateConstraint):
    r"""    
    Chance constraint:
        One probability for each time step
        conf = probability
        P(C(ti) <= bounds[ti]) > 1-conf
    """   
    
    def __init__ (self, bounds, conf=0.95):
        super().__init__ (bounds)
        assert self.bounds.ndim == 2
        self.conf = conf
        

    def num_constraints (self):
        return 1
    
    def __call__ (self, M, S, step=None, grad=False):
        C = np.zeros( 1 )

        if grad:
            dCdM = np.zeros( C.shape + M.shape )
            dCdS = np.zeros( C.shape + S.shape )
        
        P = 0;
        #Pi = np.zeros(self.num_states)
        for i in range(self.num_states):
            x1 = (self.bounds[i,0]-M[i])/(np.sqrt(S[i,i])*np.sqrt(2))
            x2 = (self.bounds[i,1]-M[i])/(np.sqrt(S[i,i])*np.sqrt(2))
            
            Pi1 = 0.5*(1+ erf(x1))
            Pi2 = 0.5*(1+ erf(x2))
            
            Pi = Pi2 - Pi1
            #print(Pi)
            
            if grad:
                dP1dM = (-1)*np.exp((-1)*(x1**2))/(np.sqrt(S[i,i])*np.sqrt(2)*np.sqrt(np.pi))
                dP2dM = (-1)*np.exp((-1)*(x2**2))/(np.sqrt(S[i,i])*np.sqrt(2)*np.sqrt(np.pi))
                dPidM = (dP2dM - dP1dM)
                dP1dS = (-1)*(self.bounds[i,0] - M[i])*np.exp((-1)*(x1**2))/(S[i,i]*np.sqrt(2)*np.sqrt(np.pi))
                dP2dS = (-1)*(self.bounds[i,1] - M[i])*np.exp((-1)*(x2**2))/(S[i,i]*np.sqrt(2)*np.sqrt(np.pi))
                dPidS = (dP2dS - dP1dS)
                if P == 1:
                    dCdM[0,i] = dPidM
                    dCdS[0,i,i] = dPidS
                else:
                    dCdM[0,i] = dCdM[0,i] + dPidM - (Pi*dCdM[0,i] + P*dPidM)
                    dCdS[0,i,i] = dCdS[0,i,i] + dPidS - (Pi*dCdS[0,i,i] + P*dPidS)
            
            if P == 1:
                P = Pi
            else:
                P = P + Pi - P*Pi
                
        C = P - self.conf
        return C if not grad else (C, dCdM, dCdS)
    
class JointChanceStateConstraint (StateConstraint):
    r"""    
    Chance constraint:
        P1(\mu_i(t)+r*S <= bounds[i,1]) > 1-eps
        P2(\mu_i(t)-r*S >= bounds[i,1]) > 1-eps
    """   
    
    def __init__ (self, bounds,conf=0.95):
        super().__init__ (bounds)
        assert self.bounds.ndim == 2
        self.conf = conf
        

    def num_constraints (self):
        return 1
    
    def __call__ (self, M, S, step=None, grad=False):
        P = np.zeros( 1 )

        if grad:
            dPdM = np.zeros( P.shape + M.shape )
            dPdS = np.zeros( P.shape + S.shape  )
        
        P = 0;
        #Pi = np.zeros(self.num_states)
        for i in range(self.num_states):
            x1 = (self.bounds[i,0]-M[i])/(np.sqrt(S[i,i])*np.sqrt(2))
            x2 = (self.bounds[i,1]-M[i])/(np.sqrt(S[i,i])*np.sqrt(2))
            
            Pi1 = 0.5*(1+ erf(x1))
            Pi2 = 0.5*(1+ erf(x2))
            
            Pi = Pi2 - Pi1
            #print(Pi)
            
            if grad:
                dP1dM = (-1)*np.exp((-1)*(x1**2))/(np.sqrt(S[i,i])*np.sqrt(2)*np.sqrt(np.pi))
                dP2dM = (-1)*np.exp((-1)*(x2**2))/(np.sqrt(S[i,i])*np.sqrt(2)*np.sqrt(np.pi))
                dPidM = (dP2dM - dP1dM)
                dP1dS = (-1)*(self.bounds[i,0] - M[i])*np.exp((-1)*(x1**2))/(S[i,i]*np.sqrt(2)*np.sqrt(np.pi))
                dP2dS = (-1)*(self.bounds[i,1] - M[i])*np.exp((-1)*(x2**2))/(S[i,i]*np.sqrt(2)*np.sqrt(np.pi))
                dPidS = (dP2dS - dP1dS)
                if P == 1:
                    dPdM[0,i] = dPidM
                    dPdS[0,i,i] = dPidS
                else:
                    dPdM[0,i] = dPdM[0,i] + dPidM - (Pi*dPdM[0,i] + P*dPidM)
                    dPdS[0,i,i] = dPdS[0,i,i] + dPidS - (Pi*dPdS[0,i,i] + P*dPidS)
            
            if P == 1:
                P = Pi
            else:
                P = P + Pi - P*Pi
                
        return P if not grad else (P, dPdM, dPdS)
