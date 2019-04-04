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
from scipy.stats import norm

class Model:
    def __init__ (self, name, num_states):
        self.name = name

        self.num_inputs = 1
        self.num_states = num_states
        self.num_meas   = 2

        # Observation matrix
        self.H = np.zeros((self.num_meas, self.num_states))
        self.H[0,0] = 1
        self.H[1,1] = 1

        # Process noise covariance
        self.Q      = 1e-6 * np.eye(self.num_states)
        self.Q[0,2] = 1e-7
        self.Q[2,0] = 1e-7

        # Measurement noise covariance
        self.R   = 2e-4 * np.eye(self.num_meas)
        # Control input covariance
        self.S_u = 1e-6 * np.eye(self.num_inputs)
        
        # Initial parameter guess
        self.p0 = [0.2, 0.1, 0.01]

        # Initial latent state
        self.x0    = np.zeros(self.num_states)
        self.x0[0] = 1
        self.x0[2] = 0.01

        # Initial latent state covariance
        self.S_x0      = np.zeros((self.num_states, self.num_states))
        self.S_x0[0,0] = 1e-4
        self.S_x0[2,2] = 1e-6
        
        # Control and latent state bounds
        self.u_bounds = np.array([[0., 0.1]])
        self.u_delta  = [ 0.01 ]
        self.x_bounds = np.array([[0.,2.],[0.,3.],[0.,5.]])

    def __call__ (self, x, u, p):
    	# Transition function
        dx = x + self.change(x, u, p)
        return np.maximum(dx, 0)

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

    def change (self, x, u, p):
        S, A, V    = x
        k1, k2, k3 = p
        dS = -k1*S*V + u[0]
        dA =  k2*S*V
        dV =  k1*S*V - k3*A*V
        return np.array([ dS, dA, dV ])


class M2 (Model):
    def __init__ (self):
        super().__init__('M2',3)
        self.p0 += [0.05, 0.01]

    def change (self, x, u, p):
        S, A, V            = x
        k1, k2, k3, k4, k5 = p
        C  = norm.cdf(S, k5, 0.05)
        dS = -C*k1*S*V + u[0]
        dA =  C*k2*S*V - (1-C)*k4*A*V
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        return np.array([ dS, dA, dV ])


class M3 (Model):
    def __init__ (self):
        super().__init__('M3',4)
        self.p0 += [0.05, 0.01, 0.01]
        self.x_bounds = np.vstack((self.x_bounds, np.array([0., 1.])))

    def change (self, x, u, p):
        S, A, V, P             = x
        k1, k2, k3, k4, k5, k6 = p
        C  = norm.cdf(S, k5, 0.1)
        dS = -C*k1*S*V + u[0] 
        dA =  C*k2*S*V - (1-C)*k4*A*V - k6*S*P
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        dP = (1-C)*0.1*k4*A*V
        return np.array([ dS, dA, dV, dP ])


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
