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
D. Espie and S. Macchietto (1989)
"The optimal design of dynamic experiments"
AIChE J 35(2): 223-229

S. P. Asprey and S. Macchietto (2000)
"Statistical tools for optimal dynamic model building"
Comp Chem Eng 24: 1261-1267

B. H. Chen and S. P. Asprey (2003)
"On the design of optimally informative dynamic experiments for model
discrimination in multiresponse nonlinear situations"
Ind Eng Chem Res 42: 1379-1390
"""

class Model:
	def __init__ (self, name):
		self.name = name

		self.num_states   = 2
		self.num_inputs   = 2
		self.num_measured = 2

		self.T  = 72.0
		self.dt = 1.5
		self.num_steps = 49

		self.H = np.eye(2)              # Observation matrix
		self.Q = np.zeros((2,2))        # Process noise covariance
		#self.R = 0.2**2 * np.eye(2)     # Measurement noise covariance
		self.R = np.array([[0.06, -0.01],[-0.01, 0.04]]) # Measurement noise covariance

		self.S_u  = np.diag([1e-6, 1e-3])
		self.x0   = np.array([ 1.0, 0.01 ])
		self.S_x0 = np.diag([1e-3, 1e-6])
		self.x0_bounds = np.array([[1, 10], [0.01, 0.01]])

		self.u_bounds = np.array([[0.05, 0.2], [5., 35.]])
		self.x_bounds = np.array([[0., 20.], [0., 30.]])

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

	@property
	def num_param (self):
		return len( self.p0 )

	@property
	def p0_covar (self):
		#return 1e-4 * np.eye( self.num_param )
		return 0.05 * np.eye( self.num_param )

	@property
	def p_bounds (self):
		return np.array([[0.01, 1.]] * self.num_param)


class M1 (Model):
    """
    Monod kinetics with constant specific death rate
    """
    def __init__ (self):
        super().__init__('M1')
        #self.p0 = np.array([ 0.3, 0.25, 0.56, 0.02 ])
        self.p0 = np.array([ 0.5, 0.5, 0.5, 0.5 ])

    def __call__ (self, x, u, p, grad=False):
        x1, x2 = x
        u1, u2 = u
        p1, p2, p3, p4 = p

        r   = p1*x2 / ( p2 + x2 )
        dx1 = (r - p4 - u1) * x1
        dx2 = -r*x1/p3 + u1*(u2 - x2)
        dM  = np.array([dx1, dx2])
        if not grad:
            return dM
        
        do = LatentStateDerivatives(self)
        
        # Gradients
        #drdx = np.array([ 0., 1]) * r * (1/x2 - 1./(p2 + x2))
        drdx = np.array([ 0., 1]) * (p1 / ( p2 + x2 ) - p1*x2 / ( p2 + x2 )**2)
        drdp = np.array([ r/p1, -r/(p2 + x2), 0., 0. ])

        d1dx = np.array([r-p4-u1, 0]) + x1*drdx 
        d2dx = np.array([-r/p3, -u1]) - x1*drdx/p3

        d1du = np.array([-x1, 0.])
        d2du = np.array([u2-x2, u1])

        d1dp = x1 * ( drdp + np.array([0., 0., 0., -1.]) )
        d2dp = ( np.array([0., 0., r/p3, 0.]) - drdp ) * x1/p3 
        
        do.dMdx = np.vstack(( d1dx, d2dx ))
        do.dMdu = np.vstack(( d1du, d2du ))
        do.dMdp = np.vstack(( d1dp, d2dp ))
        
        # Hessian
        drdpp = np.zeros(( self.num_param, self.num_param ))
        drdpp[0,1] = drdpp[1,0] = drdp[1]/p1
        drdpp[1,1] = -2 * drdp[1] / (p2 + x2)
        
        drdxx = np.zeros(( self.num_states, self.num_states ))
        #drdxx[1,1] = drdpp[1,1] + 2*drdp[1]/x2
        drdxx[1,1] = drdpp[1,1] - 2 * p1 / ( p2 + x2 )**2
        
        drdxp = np.zeros(( self.num_states, self.num_param ))
        drdxp[1,0] = drdx[1]/p1
        #drdxp[1,1] = drdpp[1,1] + drdp[1]/x2
        drdxp[1,1] = drdpp[1,1] - p1 / ( p2 + x2 )**2

        dx1dxx = np.array([[0., drdx[1]], [drdx[1], x1*drdxx[1,1]]])
        dx2dxx = -dx1dxx/p3

        dx1duu = np.zeros(( self.num_states, self.num_states ))
        dx2duu = np.array([[0., 1.], [1., 0.]])
        
        dx1dxu = np.array([[-1, 0.], [0., 0.]])
        dx2dxu = np.array([[0., 0.], [-1., 0.]])
        
        dx1dxp = np.array([ drdp + np.array([0, 0, 0, -1.]), x1 * drdxp[1] ])
        dx2dxp = np.zeros(( self.num_states, self.num_param ))
        dx2dxp[0] = -drdp/p3 + np.array([0, 0, r/p3**2, 0])
        dx2dxp[1] = -x1*drdxp[1]/p3 + np.array([0, 0, x1*drdx[1]/p3**2, 0])
        
        dx1dpp = x1 * drdpp
        dx2dpp = np.zeros(( self.num_param, self.num_param ))
        dx2dpp = -x1*drdpp/p3
        dx2dpp[2,:] = dx2dpp[:,2] = x1 * drdp / p3**2
        dx2dpp[2,2] = -2*x1*r/p3**3
        
        do.dMdxx = np.array((dx1dxx, dx2dxx))
        do.dMdxu = np.array((dx1dxu, dx2dxu))
        do.dMduu = np.array((dx1duu, dx2duu))
        do.dMdxp = np.array((dx1dxp, dx2dxp))
        do.dMdpp = np.array((dx1dpp, dx2dpp))

        return dM, do


class M2 (Model):
    """
    Contois kinetics with constant specific death rate
    """
    def __init__ (self):
        super().__init__('M2')
        #self.p0 = np.array([ 0.3, 0.03, 0.55, 0.03 ])
        self.p0 = np.array([ 0.5, 0.5, 0.5, 0.5 ])

    def __call__ (self, x, u, p, grad=False):
        x1, x2 = x
        u1, u2 = u
        p1, p2, p3, p4 = p 

        #r   = p1*x2 / ( p2*x1 + x2 )
        r   = 0. if np.all(x==0) else p1*x2 / ( p2*x1 + x2 )
        #rd  = r / ( p2*x1 + x2 )
        rd  = 0. if r == 0 else r / ( p2*x1 + x2 )
        dx1 = (r - p4 - u1) * x1
        dx2 = -r*x1/p3 + u1*(u2 - x2)
        dM  = np.array([dx1, dx2])
        if not grad:
            return dM

        do = LatentStateDerivatives(self)
        
        # Gradients
        #drdx = r * np.array([ -p2/(p2*x1+x2), 1./x2 - 1./(p2*x1+x2) ])
        drdx = np.array([ -r*p2/(p2*x1+x2), p1/( p2*x1 + x2 ) - r/(p2*x1+x2) ])
        drdp = np.array([ r/p1, -rd*x1, 0., 0. ])

        d1dx = np.array([r-p4-u1, 0]) + x1*drdx
        d2dx = np.array([-r/p3, -u1]) - x1*drdx/p3

        d1du = np.array([-x1, 0.])
        d2du = np.array([u2-x2, u1])

        d1dp = x1 * ( drdp + np.array([0., 0., 0., -1.]) )
        d2dp = ( np.array([0., 0., r/p3, 0.]) - drdp ) * x1/p3 
        
        do.dMdx = np.vstack(( d1dx, d2dx ))
        do.dMdu = np.vstack(( d1du, d2du ))
        do.dMdp = np.vstack(( d1dp, d2dp ))
        
        # Hessian
        drdpp = np.zeros(( self.num_param, self.num_param ))
        drdpp[0,1] = drdpp[1,0] = drdp[1]/p1
        drdpp[1,1] = -2 * x1 * drdp[1] / (p2*x1 + x2)
        
        drdxx = np.zeros(( self.num_states, self.num_states ))
        drdxx[0,0] = -2*p2*drdx[0]/(p2*x1+x2)
        #drdxx[0,1] = drdxx[1,0] = -drdx[0]*(2/(p2*x1+x2)-1./x2)
        drdxx[1,0] = -p1*p2 / (p2*x1+x2)**2 - 2*drdx[0]/(p2*x1+x2)
        drdxx[0,1] = drdxx[1,0]
        drdxx[1,1] = -2*drdx[1]/(p2*x1+x2)
        
        drdxp = np.zeros(( self.num_states, self.num_param ))
        drdxp[:,0] = drdx/p1
        drdxp[0,1] = -(rd*x2 + p2*drdp[1])/(p2*x1+x2)
        #drdxp[1,1] = drdp[1]*(1./x2-1./(p2*x1+x2)) + r*x1/(p2*x1+x2)**2
        drdxp[1,1] = (r-p1)*x1/(p2*x1+x2)**2 - drdp[1]/(p2*x1+x2)

        dx1dxx = np.array([[2*drdx[0], drdx[1]],[drdx[1], 0.]]) + x1*drdxx
        dx2dxx = -dx1dxx/p3

        dx1duu = np.zeros(( self.num_states, self.num_states ))
        dx2duu = np.array([[0., 1.], [1., 0.]])
        
        dx1dxu = np.array([[-1, 0.], [0., 0.]])
        dx2dxu = np.array([[0., 0.], [-1., 0.]])
        
        dx1dxp = np.array([ drdp + np.array([0, 0, 0, -1.]), [0,0,0,0] ]) + x1 * drdxp
        dx2dxp = np.zeros(( self.num_states, self.num_param ))
        dx2dxp[0] = -(drdp+x1*drdxp[0])/p3 + np.array([0, 0, -d2dx[0]/p3, 0])
        dx2dxp[1] = -x1*drdxp[1]/p3 + np.array([0, 0, x1*drdx[1]/p3**2, 0])
        
        dx1dpp = x1 * drdpp
        dx2dpp = np.zeros(( self.num_param, self.num_param ))
        dx2dpp = -x1*drdpp/p3
        dx2dpp[2,:] = dx2dpp[:,2] = x1 * drdp / p3**2 
        dx2dpp[2,2] = -2*x1*r/p3**3
        
        do.dMdxx = np.array((dx1dxx, dx2dxx))
        do.dMdxu = np.array((dx1dxu, dx2dxu))
        do.dMduu = np.array((dx1duu, dx2duu))
        do.dMdxp = np.array((dx1dxp, dx2dxp))
        do.dMdpp = np.array((dx1dpp, dx2dpp))

        return dM, do


class M3 (Model):
    """
    Linear specific growth rate
    """
    def __init__ (self):
        super().__init__('M3')
        #self.p0 = np.array([ 0.12, 0.56, 0.03 ])
        self.p0 = np.array([ 0.5, 0.5, 0.5 ])

    def __call__ (self, x, u, p, grad=False):
        x1, x2 = x
        u1, u2 = u
        p1, p2, p3 = p 

        r   = p1*x2 
        dx1 = (r - p3 - u1) * x1
        dx2 = -r*x1/p2 + u1*(u2 - x2)
        dM  = np.array([dx1, dx2])
        if not grad:
            return dM

        do = LatentStateDerivatives(self)
        
        # Gradients
        drdx = np.array([ 0, p1 ])
        drdp = np.array([ x2, 0., 0. ])

        d1dx = np.array([r-p3-u1, 0]) + x1*drdx 
        d2dx = np.array([-r/p2, -u1]) - x1*drdx/p2

        d1du = np.array([-x1, 0.])
        d2du = np.array([u2-x2, u1])

        d1dp = x1 * ( drdp - np.array([0., 0., 1]) )
        d2dp = np.array([0., r*x1/p2**2, 0.]) - drdp*x1/p2
        
        do.dMdx = np.array([ d1dx, d2dx ])
        do.dMdu = np.array([ d1du, d2du ])
        do.dMdp = np.array([ d1dp, d2dp ])
        
        # Hessian
        drdxp = np.array([[0,0,0],[1,0,0]])
        
        do.dMdxx = np.zeros((2, self.num_states, self.num_states))
        do.dMdxx[0,0,1] = do.dMdxx[0,1,0] = p1
        do.dMdxx[1,0,1] = do.dMdxx[1,1,0] = -p1/p2
        
        do.dMdxu = np.zeros((2, self.num_states, self.num_inputs))
        do.dMdxu[0,0,0] = do.dMdxu[1,1,0] = -1.
        
        do.dMduu = np.zeros((2, self.num_inputs, self.num_inputs))
        do.dMduu[1,0,1] = do.dMduu[1,1,0] = 1.
        
        dx1dxp = np.array([[x2,0,-1.],[x1,0,0]])
        dx2dxp = -x[::-1,None] * np.array([[1./p2,-p1/p2**2,0]])
        do.dMdxp = np.array((dx1dxp, dx2dxp))
        
        do.dMdpp = np.zeros((2, self.num_param, self.num_param))
        do.dMdpp[1,0,1] = do.dMdpp[1,1,0] = x1*x2/p2**2
        do.dMdpp[1,1,1] = -2*p1*x1*x2/p2**3
        
        return dM, do


class M4 (Model):
    """
    Monod kinetics with constant maintenance energy
    """
    def __init__ (self):
        super().__init__('M4')
        #self.p0 = np.array([ 0.3, 0.3, 0.55, 0.05 ])
        self.p0 = np.array([ 0.5, 0.5, 0.5 ])

    def __call__ (self, x, u, p, grad=False):
        x1, x2 = x
        u1, u2 = u
        p1, p2, p3 = p

        r   = p1*x2 / (p2 + x2)
        dx1 = (r - u1) * x1
        dx2 = -r*x1/p3 + u1*(u2 - x2)
        dM  = np.array([dx1, dx2])
        if not grad:
            return dM

        do = LatentStateDerivatives(self)
        
        # Gradients
        #drdx = r * np.array([ 0, 1/x2 - 1./(p2 + x2) ])
        drdx = np.array([ 0, p1/(p2 + x2) - r/(p2 + x2) ])
        drdp = np.array([ r/p1, -r/(p2 + x2), 0. ])

        d1dx = np.array([r-u1, 0]) + x1*drdx 
        d2dx = np.array([-r/p3, -u1]) - x1*drdx/p3

        d1du = np.array([-x1, 0.])
        d2du = np.array([u2-x2, u1])

        d1dp = x1 * drdp
        d2dp = np.array([0., 0., r*x1/p3**2]) - drdp*x1/p3 
        
        do.dMdx = np.array([ d1dx, d2dx ])
        do.dMdu = np.array([ d1du, d2du ])
        do.dMdp = np.array([ d1dp, d2dp ])
        
        # Hessian
        drdpp = np.zeros(( self.num_param, self.num_param ))
        drdpp[0,1] = drdpp[1,0] = drdp[1]/p1
        drdpp[1,1] = -2 * drdp[1] / (p2 + x2)
        
        drdxx = np.zeros(( self.num_states, self.num_states ))
        #drdxx[1,1] = drdpp[1,1] + 2*drdp[1]/x2
        drdxx[1,1] = drdpp[1,1] - 2*p1/(p2 + x2)**2
        
        drdxp = np.zeros(( self.num_states, self.num_param ))
        drdxp[1,0] = drdx[1]/p1
        #drdxp[1,1] = drdpp[1,1] + drdp[1]/x2
        drdxp[1,1] = drdpp[1,1] - p1/(p2 + x2)**2
        
        dx1dxx = np.array([[0., drdx[1]], [drdx[1], x1*drdxx[1,1]]])
        dx2dxx = -dx1dxx/p3

        dx1duu = np.zeros(( self.num_states, self.num_states ))
        dx2duu = np.array([[0., 1.], [1., 0.]])
        
        dx1dxu = np.array([[-1, 0.], [0., 0.]])
        dx2dxu = np.array([[0., 0.], [-1., 0.]])
        
        dx1dxp = np.array([ drdp, x1 * drdxp[1] ])
        dx2dxp = np.zeros(( self.num_states, self.num_param ))
        dx2dxp[0] = -drdp/p3 + np.array([0, 0, r/p3**2])
        dx2dxp[1] = -x1*drdxp[1]/p3 + np.array([0, 0, x1*drdx[1]/p3**2])
        
        dx1dpp = x1 * drdpp
        dx2dpp = np.zeros(( self.num_param, self.num_param ))
        dx2dpp = -x1*drdpp/p3
        dx2dpp[2,:] = dx2dpp[:,2] = x1 * drdp / p3**2
        dx2dpp[2,2] = -2*x1*r/p3**3
        
        do.dMdxx = np.array((dx1dxx, dx2dxx))
        do.dMdxu = np.array((dx1dxu, dx2dxu))
        do.dMduu = np.array((dx1duu, dx2duu))
        do.dMdxp = np.array((dx1dxp, dx2dxp))
        do.dMdpp = np.array((dx1dpp, dx2dpp))

        return dM, do


class DataGen (M1):
	def __init__ (self):
		super().__init__()
		self.p0     = []
		#self.p_true = np.array([ 0.31, 0.18, 0.55, 0.03 ])
		self.p_true = np.array([ 0.25, 0.25, 0.88, 0.09 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.p_true)


def get ():
	return DataGen(), [M1(), M2(), M3(), M4()]