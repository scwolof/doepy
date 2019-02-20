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
from scipy.integrate import odeint

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
	def __init__ (self):
		self.num_states   = 2
		self.num_controls = 2
		self.num_measured = 2

		self.H = np.eye(2)        # Observation matrix
		self.Q = np.zeros((2,2))  # Process noise covariance
		self.R = 0.2 * np.eye(2)  # Measurement noise covariance

		self.x0 = np.array([ 1.0, 0.01 ])
		self.x0_bounds = np.array([[1, 10], [0.01, 0.01]])

		self.T  = 72.0
		self.dt = 0.75
		self.num_steps = 97

		self.u_bounds = np.array([[0.05, 0.2], [5., 35.]])

	def __call__ (self, x, u, p):
		"""
		Inputs:
		   x   States at time t
		   u   Controls at time t
		   p   Model parameters
		Outputs:
		   dx  States at time t+1
		"""
		f = lambda y,t: self._ode_func(y,u,p)
		t = np.linspace(0, self.dt, 26)
		X = odeint(f, x, t)
		return X[-1]


class M1 (Model):
	"""
	Monod kinetics with constant specific death rate
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 0.3, 0.25, 0.56, 0.02 ])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p 

		r   = p1*x2 / ( p2 + x2 )
		dx1 = (r - p4 - u1) * x1
		dx2 = -r*x1/p3 + u1*(u2 - x2)
		return np.array([dx1, dx2])


class M2 (Model):
	"""
	Contois kinetics with constant specific death rate
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 0.3, 0.03, 0.55, 0.03 ])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p 

		r   = p1*x2 / ( p2*x1 + x2 )
		dx1 = (r - p3 - u1) * x1
		dx2 = -r*x1/p3 + u1*(u2 - x2)
		return np.array([dx1, dx2])


class M3 (Model):
	"""
	Linear specific growth rate
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 0.12, 0.56, 0.03 ])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3 = p 

		r   = p1*x2 
		dx1 = (r - p3 - u1) * x1
		dx2 = -r*x1/p2 + u1*(u2 - x2)
		return np.array([dx1, dx2])


class M4 (Model):
	"""
	Monod kinetics with constant maintenance energy
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 0.3, 0.3, 0.55, 0.05 ])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p

		r   = p1*x2 / (p2 + x2)
		dx1 = (r - u1) * x1
		dx2 = -(r/p2 + p4 )*x1 + u1*(u2 - x2)
		return np.array([dx1, dx2])



class DataGen (M1):
	def __init__ (self):
		M1.__init__(self)
		self.true_param = np.array([ 0.31, 0.18, 0.55, 0.03 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2(), M3(), M4()]