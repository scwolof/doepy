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
from ...derivatives import Derivatives
from sympy import symbols, lambdify
from numbers import Number
from .err_function import cdf_1, cdf_2

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
        self.x_bounds = np.array([[0.,1.2],[0.,3.],[0.,5.]])
        #self.z_bounds = np.array([[-0.1, 2.],[-0.1, 3.]])
        self.z_bounds = np.array([[-0.1, 1.2],[-0.1, 0.8]])
        
        self.dfdx = [[0 for i in range(self.num_states)] for j in range(self.num_states)]
        self.dfdu = [[0 for i in range(self.num_inputs)] for j in range(self.num_states)]
        self.dfdp = [[0 for i in range(self.num_param)] for j in range(self.num_states)]
        
        self.ddfdxx = [[[0 for i in range(self.num_states)] for j in range(self.num_states)] for k in range(self.num_states)]
        self.ddfdxu = [[[0 for i in range(self.num_inputs)] for j in range(self.num_states)] for k in range(self.num_states)]
        self.ddfdxp = [[[0 for i in range(self.num_param)] for j in range(self.num_states)] for k in range(self.num_states)]
        
        self.ddfduu = [[[0 for i in range(self.num_inputs)] for j in range(self.num_inputs)] for k in range(self.num_states)]
        self.ddfdux = [[[0 for i in range(self.num_states)] for j in range(self.num_inputs)] for k in range(self.num_states)]
        self.ddfdup = [[[0 for i in range(self.num_param)] for j in range(self.num_inputs)] for k in range(self.num_states)]
        
        self.ddfdpp = [[[0 for i in range(self.num_param)] for j in range(self.num_param)] for k in range(self.num_states)]
        self.ddfdpx = [[[0 for i in range(self.num_states)] for j in range(self.num_param)] for k in range(self.num_states)]
        self.ddfdpu = [[[0 for i in range(self.num_inputs)] for j in range(self.num_param)] for k in range(self.num_states)]
        
        
        self.set_grad()
        self.set_hessian()

    def __call__ (self, x, u, p, grad=False):
    	# Transition function
        
        
        if not grad:
            dx = x + self.change(x, u, p, grad)
            return np.maximum(dx, 0)
        else:
            dxi, do = self.change(x, u, p, grad)
            dx = x + dxi
            do.dMdx += np.eye(self.num_states)
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

    def set_grad(self):
        all_dev = self.get_derivatives()
        k=0;
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.dfdx[i][j] = all_dev[k]
                k+=1
        
        for i in range(self.num_states):
            for j in range(self.num_inputs):
                self.dfdu[i][j] = all_dev[k]
                k+=1
        
        for i in range(self.num_states):
            for j in range(self.num_param):
                self.dfdp[i][j] = all_dev[k]
                k+=1
                
    def set_hessian(self):
        all_dev = []
        for i in range(self.num_states):
            for j in range(self.num_states):
                all_dev = self.get_derivatives2(self.dfdx[i][j])
                for k in range(self.num_states):
                    self.ddfdxx[i][j][k] = all_dev[k]
                for k in range(self.num_inputs):
                    self.ddfdxu[i][j][k] = all_dev[k+self.num_states]
                for k in range(self.num_param):
                    self.ddfdxp[i][j][k] = all_dev[k+self.num_states+self.num_inputs]
        
        for i in range(self.num_states):
            for j in range(self.num_inputs):
                all_dev = self.get_derivatives2(self.dfdu[i][j])
                for k in range(self.num_states):
                    self.ddfdux[i][j][k] = all_dev[k]
                for k in range(self.num_inputs):
                    self.ddfduu[i][j][k] = all_dev[k+self.num_states]
                for k in range(self.num_param):
                    self.ddfdup[i][j][k] = all_dev[k+self.num_states+self.num_inputs]
                    
        for i in range(self.num_states):
            for j in range(self.num_param):
                all_dev = self.get_derivatives2(self.dfdp[i][j])
                for k in range(self.num_states):
                    self.ddfdpx[i][j][k] = all_dev[k]
                for k in range(self.num_inputs):
                    self.ddfdpu[i][j][k] = all_dev[k+self.num_states]
                for k in range(self.num_param):
                    self.ddfdpp[i][j][k] = all_dev[k+self.num_states+self.num_inputs]
                
            
        
                
    def get_grad(self, grad,x,u,p):
        jac = np.zeros((len(grad),len(grad[0])))
        for i in range(len(grad)):
            for k in range(len(grad[0])):
                jac[i][k] = grad[i][k](x,u,p)
        return jac
    
    def get_hessian(self, hessian,x,u,p):
        hes = np.zeros((len(hessian),len(hessian[0]),len(hessian[0][0])))
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                for k in range(len(hessian[0][0])):
                    if isinstance(hessian[i][j][k],Number):
                        hes[i][j][k] = hessian[i][j][k]
                    else:
                        hes[i][j][k] = hessian[i][j][k](x,u,p)
        return hes
        

    def get_derivatives(self):
        arg_symbols = symbols(['x:'+str(self.num_states),'u:'+str(self.num_inputs),'p:'+str(self.num_param)])
        sym_func = self.s_model(*arg_symbols)
        return [lambdify(arg_symbols, sy.diff(varel),'sympy')  for varsy in arg_symbols for sy in sym_func for varel in varsy ]
    
   
    def get_derivatives2(self, func):
        arg_symbols = symbols(['x:'+str(self.num_states),'u:'+str(self.num_inputs),'p:'+str(self.num_param)])
        sym_func = func(*arg_symbols)
        if isinstance(sym_func,Number):
            return [sym_func  for varsy in arg_symbols for varel in varsy ]
        else:
            return [lambdify(arg_symbols, sym_func.diff(varel))  for varsy in arg_symbols for varel in varsy ]
        
        

class M1 (Model):
    def __init__ (self):
        self.p0 = [0.2, 0.1, 0.01]
        super().__init__('M1',3)
    
    def s_model(self,x,u,p):
        S, A, V    = x
        k1, k2, k3 = p
        dS = -k1*S*V + u[0]
        dA =  k2*S*V
        dV =  k1*S*V - k3*A*V
        g = np.array([dS, dA, dV])
        return g

    def change (self, x, u, p, grad=False):
        S, A, V    = x
        k1, k2, k3 = p
        dS = -k1*S*V + u[0]
        dA =  k2*S*V
        dV =  k1*S*V - k3*A*V
        g = np.array([dS, dA, dV])
        if not grad:
            return g
        
        do = Derivatives(self.num_states, self.num_inputs, self.num_param, self.num_states)
        
        do.dMdx = self.get_grad(self.dfdx,x,u,p)
        do.dMdu = self.get_grad(self.dfdu,x,u,p)
        do.dMdp = self.get_grad(self.dfdp,x,u,p)
        
        
        do.dMdxx = self.get_hessian(self.ddfdxx,x,u,p)
        do.dMdxu = self.get_hessian(self.ddfdxu,x,u,p)
        do.dMdxp = self.get_hessian(self.ddfdxp,x,u,p)
        do.dMdux = self.get_hessian(self.ddfdux,x,u,p)
        do.dMduu = self.get_hessian(self.ddfduu,x,u,p)
        do.dMdup = self.get_hessian(self.ddfdup,x,u,p)
        do.dMdpx = self.get_hessian(self.ddfdpx,x,u,p)
        do.dMdpu = self.get_hessian(self.ddfdpu,x,u,p)
        do.dMdpp = self.get_hessian(self.ddfdpp,x,u,p)
        
        
        return g, do
        


class M2 (Model):
    def __init__ (self):
        self.p0 = [0.2, 0.1, 0.01, 0.05, 0.01]
        super().__init__('M2',3)
        #self.p0 += [0.05, 0.01]
    
    def s_model (self, x, u, p):
        S, A, V            = x
        k1, k2, k3, k4, k5 = p
        C  = cdf_1(S, k5, 0.05)
        #C=0.99
        dS = -C*k1*S*V + u[0]
        dA =  C*k2*S*V - (1-C)*k4*A*V
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        
        g = np.array([dS, dA, dV])
        return g

    def change (self, x, u, p, grad=False):
        S, A, V            = x
        k1, k2, k3, k4, k5 = p
        C  = cdf_2(S, k5**3.0, 0.05)
        #C=0.99
        dS = -C*k1*S*V + u[0]
        dA =  C*k2*S*V - (1-C)*k4*A*V
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        
        g = np.array([dS, dA, dV])
        if not grad:
            return g
        
        do = Derivatives(self.num_states, self.num_inputs, self.num_param, self.num_states)
        
        do.dMdx = self.get_grad(self.dfdx,x,u,p)
        do.dMdu = self.get_grad(self.dfdu,x,u,p)
        do.dMdp = self.get_grad(self.dfdp,x,u,p)
        
        
        do.dMdxx = self.get_hessian(self.ddfdxx,x,u,p)
        do.dMdxu = self.get_hessian(self.ddfdxu,x,u,p)
        do.dMdxp = self.get_hessian(self.ddfdxp,x,u,p)
        do.dMdux = self.get_hessian(self.ddfdux,x,u,p)
        do.dMduu = self.get_hessian(self.ddfduu,x,u,p)
        do.dMdup = self.get_hessian(self.ddfdup,x,u,p)
        do.dMdpx = self.get_hessian(self.ddfdpx,x,u,p)
        do.dMdpu = self.get_hessian(self.ddfdpu,x,u,p)
        do.dMdpp = self.get_hessian(self.ddfdpp,x,u,p)

        return g, do


class M3 (Model):
    def __init__ (self):
        self.p0 = [0.2, 0.1, 0.01, 0.05, 0.01, 0.01]
        super().__init__('M3',4)
        self.x_bounds = np.vstack((self.x_bounds, np.array([0., 0.5])))

    def s_model (self, x, u, p, grad=False):
        S, A, V, P             = x
        k1, k2, k3, k4, k5, k6 = p
        C  = cdf_1(S, k5, 0.1)
        #C = 0.98
        dS = -C*k1*S*V + u[0]
        dA =  C*k2*S*V - (1-C)*k4*A*V - k6*S*P
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        dP = (1-C)*0.1*k4*A*V
        g = np.array([dS, dA, dV, dP])
        return g
        
    def change (self, x, u, p, grad=False):
        S, A, V, P             = x
        k1, k2, k3, k4, k5, k6 = p
        C  = cdf_2(S, k5, 0.1)
        #C = 0.98
        dS = -C*k1*S*V + u[0]
        dA =  C*k2*S*V - (1-C)*k4*A*V - k6*S*P
        dV =  C*k1*S*V + (1-C)*0.5*k4*A*V - k3*A*V
        dP = (1-C)*0.1*k4*A*V
        g = np.array([dS, dA, dV, dP])
        if not grad:
            return g
        
        do = Derivatives(self.num_states, self.num_inputs, self.num_param, self.num_states)
        
        do.dMdx = self.get_grad(self.dfdx,x,u,p)
        do.dMdu = self.get_grad(self.dfdu,x,u,p)
        do.dMdp = self.get_grad(self.dfdp,x,u,p)
        
        
        do.dMdxx = self.get_hessian(self.ddfdxx,x,u,p)
        do.dMdxu = self.get_hessian(self.ddfdxu,x,u,p)
        do.dMdxp = self.get_hessian(self.ddfdxp,x,u,p)
        do.dMdux = self.get_hessian(self.ddfdux,x,u,p)
        do.dMduu = self.get_hessian(self.ddfduu,x,u,p)
        do.dMdup = self.get_hessian(self.ddfdup,x,u,p)
        do.dMdpx = self.get_hessian(self.ddfdpx,x,u,p)
        do.dMdpu = self.get_hessian(self.ddfdpu,x,u,p)
        do.dMdpp = self.get_hessian(self.ddfdpp,x,u,p)
        
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
