
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
        M   [num_models x num_meas]              Matrix of predictive means
        S   [num_models x num_meas x num_meas]   Matrix of predictive covariances
        grad                                     Return gradients wrt M and S

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
            dCdM = np.zeros((2*self.num_states, self.num_states))
            dCdS = np.zeros((2*self.num_states, self.num_states, self.num_states))
        
        for i in range(self.num_states):
            C[2*i]   = M[i] - self.bounds[i,0]
            C[2*i+1] = self.bounds[i,1] - M[i]
            if grad:
                dCdM[2*i,  i] = 1.
                dCdM[2*i+1,i] = -1.
        return C if not grad else (C, dCdM, dCdS)





