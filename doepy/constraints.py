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

class StateConstraint:
    def __init__ (self, bounds):
        """
        Input:
        value  constraint bound
        """
        self.bounds     = np.asarray( bounds )
        self.num_states = len( self.bounds ) 
        
    @property
    def has_added_variables (self):
        return False
        
    def __call__ (self, M, S, grad=False):
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
        

class MeanStateConstraint (StateConstraint):
    def __init__ (self, bounds):
        super().__init__ (bounds)
        
    def __call__ (self, M, S, grad=False):
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




class ControlConstraint:
    def __init__ (self, num_control):
        self.num_control = int( num_control )
    
class ControlDeltaConstraint (ControlConstraint):
    def __init__ (self, Delta):
        self.Delta = Delta
        super().__init__( len(self.Delta) )
    
    def __call__ (self, U, grad=False):
        N, D = U.shape
        assert D == self.num_control
        
        C = np.zeros((2*(N-1), D))
        C[:N-1] = self.Delta - ( U[1:]  - U[:-1] )
        C[N-1:] = self.Delta - ( U[:-1] - U[1:] )
        if grad:
            dCdU = np.zeros(( 2*(N-1)*D, N*D ))
            I1 = tuple( range((N-1)*D) )
            I2 = tuple( range((N-1)*D) )
            I3 = tuple( range((N-1)*D, 2*(N-1)*D) )
            I4 = tuple( range(D,N*D) )
            dCdU[( I1+I3, I2+I4 )] += 1.
            dCdU[( I2+I3, I4+I2 )] -= 1.
        return C if not grad else (C, dCdU.reshape( C.shape + U.shape ) )
