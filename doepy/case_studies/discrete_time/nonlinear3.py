
import numpy as np
from scipy.stats import norm
from ...derivatives import Derivatives

class Model:
    def __init__ (self, name, num_states):
        self.name = name

        self.num_inputs = 1
        self.num_states = num_states
        self.num_meas   = 1

        # Observation matrix
        self.H = np.zeros((self.num_meas, self.num_states))
        self.H[0,0] = 1
        
        # Process noise covariance
        self.Q      = 1e-6 * np.eye(self.num_states)
        

        # Measurement noise covariance
        self.R   = 2e-4 * np.eye(self.num_meas)
        # Control input covariance
        self.S_u = 1e-6 * np.eye(self.num_inputs)
        
        # Initial parameter guess
        self.p0 = [0.2]
        

        # Initial latent state
        self.x0    = np.zeros(self.num_states)
        
        # Initial latent state covariance
        self.S_x0      = np.zeros((self.num_states, self.num_states))
        
        
        # Control and latent state bounds
        self.u_bounds = np.array([[0., 0.1]])
        self.u_delta  = [ 0.01 ]
        self.x_bounds = np.array([0.,2.])
        self.z_bounds = np.array([-0.1, 2.])

    def __call__ (self, x, u, p, grad=False):
    	# Transition function
        
        
        if not grad:
            dx = x + self.change(x, u, p, grad)
            return np.maximum(dx, 0)
        else:
            dxi, do = self.change(x, u, p, grad)
            dx = x + dxi
            return np.maximum(dx, 0), do        
        

    @property
    def num_param (self):
        return len( self.p0 )

    @property
    def p_bounds (self):
        return np.array([[1e-8, 1.]] * self.num_param)

    def get_candidate_dict (self):
        return {'f':  self,
                'H':  self.H,
                'x0': self.x0,
                'name': self.name,
                'x_covar': self.Q,
                'y_covar': self.R,
                'u_covar': self.S_u,
                'hessian':  False,
                'x0_covar': self.S_x0,
                'x_bounds': self.x_bounds,
                'u_bounds': self.u_bounds,
                'num_meas': self.num_meas,
                'num_param': self.num_param,
                'num_inputs': self.num_inputs,
                'delta_transition': True}


class M1 (Model):
    def __init__ (self):
        super().__init__('M1',3)

    def change (self, x, u, p, grad=False):
        S = x
        k1 = p
        dS = -k1*S^2 + u[0]
      
        g = np.array([dS])
        if not grad:
            return g
        # dgdx
        dgdx    = np.zeros((1,1))
        dgdx[0] = np.array([-2*k1*S])
        # dgdu
        dgdu    = np.zeros((1,1))
        dgdu[0,0] = 1.0
        # dgdp
        dgdp = np.zeros((1,1))
        dgdp[0] = np.array([-S*S])
        # ddgddx
        ddgddx = np.zeros((1,1,1))
        ddgddx[0,0,0] = -2*k1 
                
        ddgddu = np.zeros((1,1,1))
        ddgdxu = np.zeros((1,1,1))
        
        dgddp = np.zeros((1,1,1))
        dgdup = np.zeros((1,1,1))
        dgdxp = np.zeros((1,1,1))
        
        dgdxp[0,0,0] = -V
        dgdxp[0,2,0] = -S
        dgdxp[2,0,0] = V
        dgdxp[2,2,0] = S
        
        dgdxp[1,0,1] = V
        dgdxp[1,2,1] = S
        
        dgdxp[2,1,2] = -V
        dgdxp[2,2,2] = -A
               
        
        
        do = Derivatives(self.num_states, self.num_inputs, self.num_param, self.num_states)
        do.dMdx = dgdx
        do.dMdu = dgdu
        do.dMdp = dgdp
        do.dMdxx = ddgddx
        do.dMdxu = ddgdxu
        do.dMduu = ddgddu
        
        do.dMdpp = dgddp
        do.dMdup = dgdup
        do.dMdxp = dgdxp
        
        return g, do
        


class M2 (Model):
    def __init__ (self):
        super().__init__('M2',3)
        self.p0 += [0.05, 0.01]

    def change (self, x, u, p, grad=False):
        S, A, V            = x
        k1, k2, k3, k4, k5 = p
        C  = norm.cdf(S, k5, 0.05)
        #C=0.99
        dS = -C*k1*S*V + u[0]
        dA =  C*k2*S*V - (1-C)*k4*A*V
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        
        g = np.array([dS, dA, dV])
        if not grad:
            return g
        # dgdx
        dgdx    = np.zeros((3,3))
        dgdx[0] = np.array([-C*k1*V, 0, -C*k1*S])
        dgdx[1] = np.array([C*k2*V, -(1-C)*k4*V, C*k2*S - (1-C)*k4*A])
        dgdx[2] = np.array([C*k1*V, (1-C)*0.5*k4*V - k3*V, C*k1*S + (1-C)*0.5*k4*A - k3*A])
        # dgdu
        dgdu    = np.zeros((3,1))
        dgdu[0] = 1.0
        # dgdp
        dgdp = np.zeros((3,5))
        dgdp[0] = np.array([-C*V*S, 0, 0, 0, 0])
        dgdp[1] = np.array([0, C*V*S, 0, -(1-C)*A*V, 0])
        dgdp[2] = np.array([C*V*S, 0, -A*V, 0.5*(1-C)*A*V, 0])
        # ddgddx
        ddgddx = np.zeros((3,3,3))
        ddgddx[0,2,0] = -C*k1 
        ddgddx[1,2,0] = C*k2
        ddgddx[2,2,0] = C*k1
        ddgddx[1,2,1] = -(1-C)*k4
        ddgddx[2,2,1] = (1-C)*0.5*k4 - k3
        ddgddx[0,0,2] = -C*k1
        ddgddx[1,0,2] = C*k2
        ddgddx[2,0,2] = C*k1
        ddgddx[1,1,2] = -(1-C)*k4
        ddgddx[1,2,2] = (1-C)*0.5*k4 - k3
        
        ddgddu = np.zeros((3,1,1))
        ddgdxu = np.zeros((3,3,1))
        
        dgddp = np.zeros((3,5,5))
        dgdup = np.zeros((3,1,5))
        dgdxp = np.zeros((3,3,5))
        
        dgdxp[0,0,0] = -C*V
        dgdxp[2,0,0] = C*V
        dgdxp[0,2,0] = C*S
        dgdxp[2,2,0] = C*S
        
        dgdxp[1,0,1] = C*V
        dgdxp[1,2,1] = C*S
        
        dgdxp[2,1,2] = -V
        dgdxp[2,2,2] = -A
        
        dgdxp[1,1,3] = -(1-C)*V
        dgdxp[2,1,3] = (1-C)*V*0.5
        dgdxp[1,2,3] = -(1-C)*A
        dgdxp[2,2,3] = (1-C)*A*0.5
        
        
        do = Derivatives(self.num_states, self.num_inputs, self.num_param, self.num_states)
        do.dMdx = dgdx
        do.dMdu = dgdu
        do.dMdp = dgdp
        do.dMdxx = ddgddx
        do.dMdxu = ddgdxu
        do.dMduu = ddgddu
        
        do.dMdpp = dgddp
        do.dMdup = dgdup
        do.dMdxp = dgdxp
        return g, do


class M3 (Model):
    def __init__ (self):
        super().__init__('M3',4)
        self.p0 += [0.05, 0.01, 0.01]
        self.x_bounds = np.vstack((self.x_bounds, np.array([0., 0.5])))

    def change (self, x, u, p, grad=False):
        S, A, V, P             = x
        k1, k2, k3, k4, k5, k6 = p
        C  = norm.cdf(S, k5, 0.1)
        #C = 0.99
        dS = -C*k1*S*V + u[0] 
        dA =  C*k2*S*V - (1-C)*k4*A*V - k6*S*P
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        dP = (1-C)*0.1*k4*A*V
        g = np.array([dS, dA, dV, dP])
        if not grad:
            return g
        # dgdx
        dgdx    = np.zeros((4,4))
        dgdx[0] = np.array([-C*k1*V, 0, -C*k1*S, 0])
        dgdx[1] = np.array([C*k2*V - k6*P, -(1-C)*k4*V, C*k2*S - (1-C)*k4*A, -k6*S])
        dgdx[2] = np.array([C*k1*V, (1-C)*0.5*k4*V - k3*V, C*k1*S + (1-C)*0.5*k4*A - k3*A, 0])
        dgdx[3] = np.array([0, (1-C)*0.1*k4*V, (1-C)*0.1*k4*A, 0])
        # dgdu
        dgdu    = np.zeros((4,1))
        dgdu[0] = 1.0
        # dgdp
        dgdp    = np.zeros((4,6))
        dgdp[0,0] = -C*V*S
        dgdp[1] = np.array([0, C*V*S, 0, -(1-C)*A*V, 0, -S*P])
        dgdp[2] = np.array([C*V*S, 0, -A*V, 0.5*(1-C)*A*V, 0, 0])
        dgdp[3] = np.array([0, 0, 0, 0.1*(1-C)*A*V, 0, 0])
        
        # ddgddx
        ddgddx = np.zeros((4,4,4))
        ddgddx[0,2,0] = -C*k1 
        ddgddx[1,2,0] = C*k2
        ddgddx[1,3,0] = -k6
        ddgddx[2,2,0] = C*k1
        
        ddgddx[1,2,1] = -(1-C)*k4
        ddgddx[2,2,1] = (1-C)*0.5*k4 - k3
        ddgddx[3,2,1] = (1-C)*0.1*k4
                
        ddgddx[0,0,2] = -C*k1
        ddgddx[1,0,2] = C*k2
        ddgddx[2,0,2] = C*k1
        ddgddx[1,1,2] = -(1-C)*k4
        ddgddx[2,1,2] = (1-C)*0.5*k4 - k3
        ddgddx[3,1,2] = (1-C)*0.1*k4
        
        ddgddx[1,0,3] = -k6
        
        ddgddu = np.zeros((4,1,1))
        ddgdxu = np.zeros((4,4,1))
        
        dgddp = np.zeros((4,6,6))
        dgdup = np.zeros((4,1,6))
        dgdxp = np.zeros((4,4,6))
        
        dgdxp[0,0,0] = -C*V
        dgdxp[2,0,0] = C*V
        dgdxp[0,2,0] = -C*S
        dgdxp[2,2,0] = C*S
        
        dgdxp[1,0,1] = C*V
        dgdxp[1,2,1] = C*S
        
        dgdxp[2,1,2] = -V
        dgdxp[2,2,2] = -A
        
        dgdxp[1,1,3] = -(1-C)*V
        dgdxp[2,1,3] = 0.5*(1-C)*V
        dgdxp[3,1,3] = 0.1*(1-C)*V
        dgdxp[1,2,3] = -(1-C)*A
        dgdxp[2,2,3] = 0.5*(1-C)*A
        dgdxp[3,2,3] = 0.1*(1-C)*A
        
        dgdxp[1,0,5] = -P
        dgdxp[1,3,5] = -S       
        
        
        do = Derivatives(self.num_states, self.num_inputs, self.num_param, self.num_states)
        do.dMdx = dgdx
        do.dMdu = dgdu
        do.dMdp = dgdp
        do.dMdxx = ddgddx
        do.dMdxu = ddgdxu
        do.dMduu = ddgddu
        
        do.dMdpp = dgddp
        do.dMdup = dgdup
        do.dMdxp = dgdxp
        return g, do


class DataGen (M2):
    def __init__ (self):
        super().__init__()

    @property
    def p (self):
        return [0.201, 0.098, 0.0099, 0.051, 0.0101] #, 0.0098]

    def __call__ (self, x, u):
        return super().__call__(x, u, self.p)

    def get_initial_experiments (self, num_exp, num_steps, u_var=0.02):
        ones  = lambda n: np.ones(( n, self.num_inputs ))
        zeros = lambda n: np.zeros(( n, self.num_inputs ))

        U_data = [ zeros(num_steps) ]

        for n in range(num_exp-1):
            r  = u_var * np.random.rand( self.num_inputs )
            U1 = r[None,:] * ones(num_steps)
            for n in np.arange(num_steps)[::-1]:
                umax  = ( num_steps - n - 1 ) * np.array(self.u_delta)
                U1[n] = np.minimum( U1[n], umax )
            U_data.append( U1 )

        return U_data

    def get_candidate_dict (self):
        d = super().get_candidate_dict()
        d['num_param'] = 0
        return d
    
def get ():
    return DataGen(), [M1(), M2(), M3()]
