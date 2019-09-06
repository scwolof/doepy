# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:35:44 2019

@author: edusc_000
"""

import numpy as np

from ..derivatives import LatentStateDerivatives
from sympy import symbols, lambdify
from numbers import Number

"""
Penicillin case study
"""

class Model:
    def __init__ (self, name, grad=True):
        self.name = name


        self.num_states   = 4
        self.num_inputs   = 4
        self.num_measured = 4

        self.H = np.eye(4)           # Observation matrix
        self.Q = np.zeros((4,4))     # Process noise covariance
        self.R = 0 * np.eye(4)    # Measurement noise covariance

        self.S_u  = 0 * np.eye(self.num_inputs)
        self.x0   = np.array([ 1, 0.2, 0.001, 250])
        #self.S_x0 = 1e-2 * np.eye(self.num_states)
        self.S_x0 = 0*np.diag(self.x0/1e4)
        self.x0_bounds = np.array([[None, None], [None, None],[None, None],[None, None]])
        #self.p0_covar = np.diag(self.p0/1e4)

        self.T  = 40
        self.dt = 1.
        self.num_steps = 40

        self.u_bounds = np.array([[0, 10],[-1, 1],[-1, 1],[-1, 1]])
        self.x_bounds = np.array([[None, None], [None, None],[None, None],[None, None]])
        self.z_bounds = np.array([[None, None], [None, 0.49],[None, None],[None, None]])
        
        if grad:
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

    def get_candidate_dict (self):
        return {'f':  self,
                'H':  self.H,
                'x0': self.x0,
                'name': self.name,
                'x_covar': self.Q,
                'u_covar': self.S_u,
                'y_covar': self.R,
                'hessian': False,
                'x0_covar':  self.S_x0,
                'x_bounds':  self.x_bounds,
                'u_bounds':  self.u_bounds,
                'num_meas':  self.num_measured,
                'num_param': self.num_param,
                'num_inputs':  self.num_inputs,
                'step_length': self.dt}

    @property
    def num_param (self):
        return len( self.p0 )

    @property
    def p0_covar (self):
        #return np.diag(self.p0/1e3)
        return 0. * np.eye( self.num_param )

    @property
    def p_bounds (self):
        return np.array([[0., 1.]] * self.num_param)
    
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
                
            
        
                
    def get_grad(self, grad,t,x,u,p):
        jac = np.zeros((len(grad),len(grad[0])))
        for i in range(len(grad)):
            for k in range(len(grad[0])):
                jac[i][k] = grad[i][k](t,x,u,p)
        return jac
    
    def get_hessian(self, hessian,t,x,u,p):
        hes = np.zeros((len(hessian),len(hessian[0]),len(hessian[0][0])))
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                for k in range(len(hessian[0][0])):
                    if isinstance(hessian[i][j][k],Number):
                        hes[i][j][k] = hessian[i][j][k]
                    else:
                        hes[i][j][k] = hessian[i][j][k](t,x,u,p)
        return hes
        

    def get_derivatives(self):
        arg_symbols = symbols(['t:'+str(1),'x:'+str(self.num_states), 'u:'+str(self.num_inputs),'p:'+str(self.num_param)])
        sym_func = self.s_model(*arg_symbols)
        for k, sy in enumerate(sym_func):
            if isinstance(sy,Number):
                sy = symbols('kk')
                sym_func[k] = sy
        return [lambdify(arg_symbols, sy.diff(varel),'sympy')  for varsy in arg_symbols for sy in sym_func for varel in varsy if varel not in arg_symbols[0]]
    
   
    def get_derivatives2(self, func):
        arg_symbols = symbols(['t:'+str(1),'x:'+str(self.num_states),'u:'+str(self.num_inputs),'p:'+str(self.num_param)])
        sym_func = func(*arg_symbols)
        if isinstance(sym_func,Number):
            return [0  for varsy in arg_symbols for varel in varsy if varel not in arg_symbols[0]]
        else:
            return [lambdify(arg_symbols, sym_func.diff(varel))  for varsy in arg_symbols for varel in varsy if varel not in arg_symbols[0]]


class M1 (Model):
    """
    Monod kinetics with constant specific death rate
    """
    def __init__ (self, grad=True):
        Kl = 6e-3;
        mi = 0.11;
        Yxs = 0.47;
        teta = 4e-3;
        Yp = 1.2;
        Ki = 0.1;
        Mx = 0.029;
        Kxp = 0.01;
        Kp = 1e-4;
        S0 = 400;

        self.p0 = np.array([Kl, mi, Yxs, teta, Yp, Ki, Mx, Kxp, Kp, S0])
        #self.p0 = np.array([Kl, Yxs, Yp, Ki, Mx, Kxp, Kp, S0])
        super().__init__('M1',grad)
        
    def s_model(self,t,x,u,p):
        x1, x2, x3, x4 = x
        #u1 = u[0]
        Kl, mi, Yxs, teta, Yp, Ki, Mx, Kxp, Kp, S0 = p
        #Kl, Yxs, Yp, Ki, Mx, Kxp, Kp, S0 = p
        #mi = x5
        #teta = x6
        t = t[0]
        u1, u2, u3, u4 = u
        ut = u1 + u2*t + u3*(t**2) + u4*(t**3)

        dx1 = mi*x1*x2/(Kl*x1+x2) - ut*x1/x4
        dx2 = -mi*x1*x2/((Kl*x1+x2)*Yxs) - Mx*x1 - teta*x1*x2/(Yp*(x2+Kp+x2**2/Ki)) + ut*(S0-x2)/x4
        dx3 = teta*x1*x2/(x2+Kp+x2**2/Ki) - Kxp*x3 - ut*x3/x4
        dx4 = ut

        dM  = np.array([dx1, dx2, dx3, dx4])
        return dM
    
    def __call__ (self, t, x, u, p, grad=False, param_uncertainty=True):
        x1, x2, x3, x4 = x
        #print(x)
        #u1 = u[0]
        Kl, mi, Yxs, teta, Yp, Ki, Mx, Kxp, Kp, S0 = p
        #Kl, Yxs, Yp, Ki, Mx, Kxp, Kp, S0 = p
        
        #mi = x5
        #teta = x6
        #t = t[0]
        

        u1, u2, u3, u4 = u
        ut = u1 + u2*t + u3*(t**2) + u4*(t**3)
        dx1 = mi*x1*x2/(Kl*x1+x2) - ut*x1/x4
        dx2 = -mi*x1*x2/((Kl*x1+x2)*Yxs) - Mx*x1 - teta*x1*x2/(Yp*(x2+Kp+x2**2/Ki)) + ut*(S0-x2)/x4
        dx3 = teta*x1*x2/(x2+Kp+x2**2/Ki) - Kxp*x3 - ut*x3/x4
        dx4 = ut

        dM  = np.array([dx1, dx2, dx3, dx4])
        if not grad:
            return dM
        
        do = LatentStateDerivatives(self)
        t = [t]
        
        do.dMdx = self.get_grad(self.dfdx,t,x,u,p)
        do.dMdu = self.get_grad(self.dfdu,t,x,u,p)
        
        if param_uncertainty:        
            do.dMdp = self.get_grad(self.dfdp,t,x,u,p)
            do.dMdxx = self.get_hessian(self.ddfdxx,t,x,u,p)
            do.dMdxu = self.get_hessian(self.ddfdxu,t,x,u,p)
            do.dMdxp = self.get_hessian(self.ddfdxp,t,x,u,p)
            do.dMdux = self.get_hessian(self.ddfdux,t,x,u,p)
            do.dMduu = self.get_hessian(self.ddfduu,t,x,u,p)
            do.dMdup = self.get_hessian(self.ddfdup,t,x,u,p)
            do.dMdpx = self.get_hessian(self.ddfdpx,t,x,u,p)
            do.dMdpu = self.get_hessian(self.ddfdpu,t,x,u,p)
            do.dMdpp = self.get_hessian(self.ddfdpp,t,x,u,p)

        return dM, do


def get ():
    return [M1()]