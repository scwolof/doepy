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

		self.H = np.eye(2)           # Observation matrix
		self.Q = np.zeros((2,2))     # Process noise covariance
		self.R = 0.2**2 * np.eye(2)  # Measurement noise covariance

		self.S_u  = 1e-6 * np.eye(self.num_inputs)
		self.x0   = np.array([ 1.0, 0.01 ])
		self.S_x0 = 1e-6 * np.eye(self.num_states)
		self.x0_bounds = np.array([[1, 10], [0.01, 0.01]])

		self.T  = 72.0
		self.dt = 0.75
		self.num_steps = 97

		self.u_bounds = np.array([[0.05, 0.2], [5., 35.]])
		self.x_bounds = np.array([[0., 20.], [0., 30.]])

	@property
	def num_param (self):
		return len( self.p0 )

	@property
	def S_p (self):
		return 1e-4 * np.eye(self.num_param)

	def get_candidate_dict (self):
		return {'f':  self,
		        'H':  self.H,
		        'x0': self.x0,
		        'name': self.name,
		        'x_covar':  self.Q,
		        'u_covar':  self.S_u,
		        'p_covar':  self.S_p,
		        'y_covar':  self.R,
		        'hessian':  False,
		        'x0_covar': self.S_x0,
		        'x_bounds': self.x_bounds,
		        'u_bounds': self.u_bounds,
		        'num_meas': self.num_meas,
		        'num_inputs': self.num_inputs}


class M1 (Model):
	"""
	Monod kinetics with constant specific death rate
	"""
	def __init__ (self):
		super().__init__('M1')
		self.p0 = np.array([ 0.3, 0.25, 0.56, 0.02 ])

	def __call__ (self, x, u, p, grad=False):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p 

		r   = p1*x2 / ( p2 + x2 )
		dx1 = (r - p4 - u1) * x1
		dx2 = -r*x1/p3 + u1*(u2 - x2)
		dx  = np.array([dx1, dx2])
		if not grad:
			return dx

		drdx = np.array([ 0., 1]) * r * (1/x2 - 1./(p2 + x2))
		drdp = np.array([ r/p1, -r/(p2 + x2), 0., 0. ])

		d1dx = np.array([r-p4-u1, 0]) + x1*drdx 
		d2dx = np.array([-r/p3, -u1]) - x1*drdx/p3
		dxdx = np.vstack(( d1dx, d2dx ))

		d1du = np.array([-x1, 0.])
		d2du = np.array([u2-x2, u1])
		dxdu = np.vstack(( d1du, d2du ))

		d1dp = x1 * ( drdp + np.array([0., 0., 0., -1.]) )
		d2dp = ( np.array([0., 0., r/p3, 0.]) - drdp ) * x1/p3 
		dxdp = np.vstack(( d1dp, d2dp ))

		return dx, dxdx, dxdu, dxdp


class M2 (Model):
	"""
	Contois kinetics with constant specific death rate
	"""
	def __init__ (self):
		super().__init__('M2')
		self.p0 = np.array([ 0.3, 0.03, 0.55, 0.03 ])

	def __call__ (self, x, u, p, grad=False):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p 

		r   = p1*x2 / ( p2*x1 + x2 )
		dx1 = (r - p4 - u1) * x1
		dx2 = -r*x1/p3 + u1*(u2 - x2)
		dx  = np.array([dx1, dx2])
		if not grad:
			return dx

		drdx = r * np.array([ -p2/(p2*x1 + x2), 1/x2 - 1./(p2*x1 + x2) ])
		drdp = np.array([ r/p1, -r*x1/(p2*x1 + x2), 0., 0. ])

		d1dx = np.array([r-p4-u1, 0]) + x1*drdx 
		d2dx = np.array([-r/p3, -u1]) - x1*drdx/p3
		dxdx = np.vstack(( d1dx, d2dx ))

		d1du = np.array([-x1, 0.])
		d2du = np.array([u2-x2, u1])
		dxdu = np.vstack(( d1du, d2du ))

		d1dp = x1 * ( drdp + np.array([0., 0., 0., -1.]) )
		d2dp = ( np.array([0., 0., r/p3, 0.]) - drdp ) * x1/p3 
		dxdp = np.vstack(( d1dp, d2dp ))

		return dx, dxdx, dxdu, dxdp


class M3 (Model):
	"""
	Linear specific growth rate
	"""
	def __init__ (self):
		super().__init__('M3')
		self.p0 = np.array([ 0.12, 0.56, 0.03 ])

	def __call__ (self, x, u, p, grad=False):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3 = p 

		r   = p1*x2 
		dx1 = (r - p3 - u1) * x1
		dx2 = -r*x1/p2 + u1*(u2 - x2)
		dx  = np.array([dx1, dx2])
		if not grad:
			return dx

		drdx = np.array([ 0, p1 ])
		drdp = np.array([ x2, 0., 0. ])

		d1dx = np.array([r-p3-u1, 0]) + x1*drdx 
		d2dx = np.array([-r/p2, -u1]) - x1*drdx/p2
		dxdx = np.vstack(( d1dx, d2dx ))

		d1du = np.array([-x1, 0.])
		d2du = np.array([u2-x2, u1])
		dxdu = np.vstack(( d1du, d2du ))

		d1dp = x1 * ( drdp - np.array([0., 0., 1]) )
		d2dp = np.array([0., r*x1/p2**2, 0.]) - drdp*x1/p2
		dxdp = np.vstack(( d1dp, d2dp ))

		return dx, dxdx, dxdu, dxdp


class M4 (Model):
	"""
	Monod kinetics with constant maintenance energy
	"""
	def __init__ (self):
		super().__init__('M4')
		self.p0 = np.array([ 0.3, 0.3, 0.55, 0.05 ])

	def __call__ (self, x, u, p, grad=False):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p

		r   = p1*x2 / (p2 + x2)
		dx1 = (r - u1) * x1
		dx2 = -(r/p3 + p4 )*x1 + u1*(u2 - x2)
		dx  = np.array([dx1, dx2])
		if not grad:
			return dx

		drdx = r * np.array([ 0, 1/x2 - 1./(p2 + x2) ])
		drdp = np.array([ r/p1, -r/(p2 + x2), 0., 0. ])

		d1dx = np.array([r-u1, 0]) + x1*drdx 
		d2dx = np.array([-(r/p3+p4), -u1]) - x1*drdx/p3
		dxdx = np.vstack(( d1dx, d2dx ))

		d1du = np.array([-x1, 0.])
		d2du = np.array([u2-x2, u1])
		dxdu = np.vstack(( d1du, d2du ))

		d1dp = x1 * drdp
		d2dp = np.array([0., 0., r*x1/p3**2, -x1]) - drdp*x1/p3 
		dxdp = np.vstack(( d1dp, d2dp ))

		return dx, dxdx, dxdu, dxdp



class DataGen (M1):
	def __init__ (self):
		super().__init__()
		self.true_param = np.array([ 0.31, 0.18, 0.55, 0.03 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2(), M3(), M4()]