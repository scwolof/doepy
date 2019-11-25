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

from ..derivatives import LatentStateDerivatives

"""
P. Bania (2019)
"Bayesian input design for linear dynamical model discrimination"
Entropy 21(4):351
"""

class Model:
    def __init__ (self, name, num_states):
        self.name = name
        self.num_states   = num_states
        self.num_inputs   = 1
        self.num_measured = 1
        self.num_param    = 0

        self.T  = 20.0
        self.num_steps = 25 #int( self.T / self.dt )
        self.dt = self.T / self.num_steps

        # Observation matrix
        self.H = np.zeros((1, self.num_states))  
        self.H[0,0] = 1

        # Process noise covariance
        self.Q = np.zeros((self.num_states,self.num_states))
        self.Q[-1,-1] = 0.05**2

        # Measurement noise covariance
        self.R = np.array([[0.05**2]])

        self.S_u  = np.array([[0]])
        self.x0   = np.zeros(self.num_states)       
        self.S_x0 = np.zeros((self.num_states, self.num_states))
        self.x0_bounds = np.zeros((self.num_states, 2))

        self.u_bounds = np.array([[-1, 1]])
        self.x_bounds = np.array([[-2, 2], [-2, 2]])

    def get_candidate_dict (self):
        return {'f':  self,
                'H':  self.H,
                'x0': self.x0,
                'name': self.name,
                'x_covar': self.Q,
                'u_covar': self.S_u,
                'y_covar': self.R,
                'hessian': True,
                'x0_covar':  self.S_x0,
                'x_bounds':  self.x_bounds,
                'u_bounds':  self.u_bounds,
                'num_meas':  self.num_measured,
                'num_param': self.num_param,
                'num_inputs':  self.num_inputs,
                'step_length': self.dt}

    def __call__ (self, x, u, grad=False):
        f = np.matmul(self.A, x) + self.B * u
        if not grad:
            return f
        
        do = LatentStateDerivatives(self)
        do.dMdx = self.A
        do.dMdu = self.B.reshape((self.num_states, 1))

        return f, do


class M1 (Model):
    def __init__ (self):
        super().__init__('M1', 1)
        self.A = np.array([[-1]])
        self.B = np.array([1])


class M2 (Model):
    def __init__ (self):
        super().__init__('M2', 2)
        self.A = np.array([[0, 1],[-3, -2.5]])
        self.B = np.array([0, 3])


class M3 (Model):
    def __init__ (self):
        super().__init__('M3', 3)
        self.A = np.array([[0,1,0],[-3,-3.5,1],[0,0,-10]])
        self.B = np.array([0,0,30])


class DataGen (M2):
    def __init__ (self):
        super().__init__()

    def __call__ (self, x, u):
        return super().__call__(x, u)


def get ():
    return DataGen(), [M1(), M2(), M3()]