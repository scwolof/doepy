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
D. Skanda and D. Lebiedz (2010)
"An optimal experimental design approach to model discrimination
in dynamic biochemical systems"
Bioinformatics 26(7): 939-945

D. Skanda and D. Lebiedz (2013)
"A robust optimization approach to experimental design for model 
discrimination of dynamical systems"
Math Program, Series A 141: 405-433

Comments:
Original case study does not have control inputs.
The experimental design is wrt to measurement time points.
We modify this by introducing controls u1 and u2 that perturb the states.
"""

class Model:
	def __init__ (self, name):
		"""
		Inputs:
		   x    States at time t:
		        x[0] - substrate concentration
		        x[1] - product concentration
		   u    Controls at time t
		   p    Model parameters
		"""
		self.name = name 

		self.num_states   = 2
		self.num_controls = 2
		self.num_measured = 2

		self.H = np.eye(2)        # Observation matrix
		self.Q = np.zeros((2,2))  # Process noise covariance
		self.R = np.eye(2)        # Measurement noise covariance

		self.x0 = np.array([ 15., 2. ])

		self.T  = 300.0
		self.dt = 10
		self.num_steps = 31

		self.u_bounds = np.array([[0., 0.2], [0., 0.2]])
		self.x_bounds = np.array([[1e-7, 25],[1e-7, 25]])

	@property
	def num_param (self):
		return len( self.p0 )

	@property
	def S_p (self):
		return 1e-4 * np.eye( self.num_param )


class M1 (Model):
	"""
	Allosteric model with positive feedback and linear product sink
	"""
	def __init__ (self):
		super().__init__('M1')
		self.p0 = np.array([ 0.879, 2.03, 0.109, 1.61e4 ])
		self.p_bounds = np.array([[1e-7, 10],[1e-7,10],[1e-7,10],[1e3,1e5]])

	def __call__ (self, x, u, p):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p

		r1  = x1 * (1+x1) * (1+x2)**2
		r2  = p4 + (1+x1)**2 * (1+x2)**2
		r   = r1 / r2
		dx1 = 0.22 - p1 * r + u1
		dx2 = (p1 * p2 * r) - (p3 * x2) + u2
		dx  = np.array([dx1, dx2])
		if not grad:
			return dx

		dr1dx = np.array([ (1+2*x1)*(1+x2), 2*x1*(1+x1) ]) * (1+x2)
		dr2dx = np.array([ 1+x2, 1+x1 ]) * 2 * (1+x1) * (1+x2)
		drdx  = ( dr1dx*r2 - r1*dr2dx ) / r2**2
		dx1dx = -p1 * drdx
		dx2dx =  p1 * p2 * drdx - np.array([0,p3])
		dxdx  = np.vstack((dx1dx, dx2dx))

		dxdu  = np.eye(2)

		drdp  = np.array([0,0,0, -r1/r2**2])
		dx1dp = np.array([-r,0,0,0]) - p1 * drdp
		dx2dp = np.array([p2*r,p1*r,-x2,0]) + p1 * p2 * drdp
		dxdp  = np.vstack((dx1dp, dx2dp))

		return dx, dxdx, dxdu, dxdp


class M2 (Model):
	"""
	Allosteric model with positive feedback in the absence of cooperativity
	when the product sink is represented by Michaelis-Menten kinetics
	"""
	def __init__ (self):
		super().__init__('M2')
		self.p0 = np.array([ 6.47, 3.33, 5.11, 244 ])
		self.p_bounds = np.array([[1e-7, 100],[1e-7,100],[1e-7,100],[100,300]])

	def __call__ (self, x, u, p):
		x1, x2 = x
		u1, u2 = u
		p1, p2, p3, p4 = p 

		r1  = x1 * (1+x2)
		r2  = p4 + (1+x1) * (1+x2)
		r   = r1 / r2
		dx1 = 0.22 - r + u1
		dx2 = p1 * r - (p2 * x2) / (p3 + x2) + u2
		dx  = np.array([dx1, dx2])
		if not grad:
			return dx

		dr1dx = np.array([ 1+x2, x1 ])
		dr2dx = np.array([ 1+x2, 1+x1 ])
		drdx  = ( dr1dx*r2 - r1*dr2dx ) / r2**2
		dx1dx = -drdx
		dx2dx =  p1 * drdx + np.array([0, p2*(x2/(p3 + x2)**2 - 1./(p3 + x2))])
		dxdx  = np.vstack((dx1dx, dx2dx))

		dxdu  = np.eye(2)

		drdp  = np.array([0,0,0, -r1/r2**2])
		dx1dp = -drdp
		dx2dp = p1*drdp + np.array([0,-x2,p2*x2/(p3 + x2),0])/ (p3 + x2)
		dxdp  = np.vstack((dx1dp, dx2dp))

		return dx, dxdx, dxdu, dxdp


class DataGen (M1):
	def __init__ (self):
		M1.__init__(self)
		self.true_param = np.array([ 0.889, 2.16, 0.111, 1.76e4 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2()]