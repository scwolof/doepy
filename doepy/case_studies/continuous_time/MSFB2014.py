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
A. Mesbah, S. Streif, R. Findeisen and R. D. Braatz (2014)
"Active fault diagnosis for nonlinear systems with probabilistic uncertainties"
IFAC Proceedings (2014): 7079-7084
"""

class Model:
	def __init__ (self, name):
		"""
		Inputs:
		   x    States at time t:   x[i] - water level tank i (m)
		   u    Controls at time t: u[i] - water inflow (m^3 / s)
		   p    Model parameters
		"""
		self.name = name

		self.num_states   = 3
		self.num_controls = 2
		self.num_measured = 1

		self.H = np.array([0, 0, 1.])  # Observation matrix
		self.Q = np.zeros((3, 3))      # Process noise covariance
		self.R = 0.01**2 * np.eye(1)   # Measurement noise covariance

		self.x0 = np.array([ 0., 0., 0. ])
		self.P0 = np.zeros(( 3, 3 ))
		self.x_bounds = np.array([[0., 0.75], [0., 0.75], [0., 0.75]])
		self.u_bounds = np.array([[0., 1e-4], [0., 1e-4]])

		self.T  = 3000
		self.dt = 50
		self.num_steps = int( self.T / self.dt )

		self.A  = 0.0154  # Tank cross-section area (m^2)
		self.Sp = 5e-5    # Connecting pipe cross-section area (m^2)
		self.g  = 9.81    # Gravity constant (m/s^2)

	def _t_dx (self, x, p, grad=False):
		s2g = p[:3] * np.sqrt( 2 * self.g )
		js1 = np.array([[1,0,-1],[0,1,0],[0,-1,1]])
		xs  = np.matmul(js1, x)
		tt  = s2g * np.sign( xs )
		t   = tt * np.sqrt(np.abs( xs ))
		js2 = np.array([[-1,0.,0.],[0.,-1,1],[1,0.,-1]])
		dx  = self.Sp * np.matmul(js2, t)
		if not grad:
			return dx, t

		dtdp = np.diag(t / p[:3])
		dtdx = 0.5 * (s2g / np.sqrt(np.abs( xs )))[:,None] * js1
		dxdx = self.Sp * np.matmul(js2, dtdx)
		dxdp = self.Sp * np.matmul(js2, dtdp)

		return dx, t, dxdx, dxdp, dtdx, dtdp

	@property
	def num_param (self):
		return len( self.p0 )

	@property
	def S_p (self):
		return 1e-4 * np.eye( self.num_param )


class M1 (Model):
	"""
	Nominal scenario (no fault)
	"""
	def __init__ (self):
		Model.__init__(self, 'M1')
		self.p0 = np.array([ 1., 0.8, 1. ])
		self.p_bounds  = np.array([[0.9, 1.1], [0.7, 0.9], [0.9, 1.1]])

	def __call__ (self, x, u, p, grad=False):
		res = self._t_dx(x, p, grad=grad)
		dx  = ( res[0] + np.array([ u[0], u[1], 0 ]) ) / self.A
		if not grad:
			return dx

		dxdx, dxdp = [ r/self.A for r in res[2:4] ]
		dxdu = np.array([[1.,0],[0,1.],[0,0]]) / self.A
		return dx, dxdx, dxdu, dxdp


class M2 (Model):
	"""
	Multiplicative actuator fault in inlet pump
	"""
	def __init__ (self):
		Model.__init__(self, 'M2')
		self.p0 = np.array([ 1., 0.8, 1., 0.6 ])
		self.p_bounds  = np.array([[0.9, 1.1],[0.7, 0.9],[0.9, 1.1],[0.5,0.7]])

	def __call__ (self, x, u, p, grad=False):
		res = self._t_dx(x, p, grad=grad)
		dx  = ( res[0] + np.array([ p[3]*u[0], u[1], 0 ]) ) / self.A
		if not grad:
			return dx

		dxdx, dxdp = [ r/self.A for r in res[2:4] ]
		dxdu  = np.array([[p[3],0],[0,1.],[0,0]]) / self.A
		dxdp4 = np.array([u[0]/self.A,0,0])
		return dx, dxdx, dxdu, np.c_[dxdp, dxdp4]


class M3 (Model):
	"""
	Circular leak in tank
	"""
	def __init__ (self):
		Model.__init__(self, 'M3')
		self.p0 = np.array([ 1., 0.8, 1., 0.002 ])
		self.p_bounds  = np.array([[0.9, 1.1],[0.7, 0.9],[0.9, 1.1],[0., 0.004]])

	def __call__ (self, x, u, p, grad=False):
		res  = self._t_dx(x, p, grad=grad)
		leak = res[1][1] * np.pi * p[3]**2
		dx   = ( res[0] + np.array([ u[0], u[1] - leak, 0 ]) ) / self.A
		if not grad:
			return dx
        
		dxdx, dxdp, dtdx, dtdp = [ r/self.A for r in res[2:] ]
		dxdx[1] -= dtdx[1] * np.pi * p[3]**2
		dxdp[1] -= dtdp[1] * np.pi * p[3]**2
		dxdu     = np.array([[1.,0],[0,1.],[0,0]]) / self.A
		dxdp4    = np.array([0,-2*leak/p[3],0]) / self.A
		return dx, dxdx, dxdu, np.c_[dxdp, dxdp4]


class DataGen (M2):
	def __init__ (self):
		M1.__init__(self)
		self.true_param = np.array([ 0.97, 0.82, 0.96, 0.67 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2(), M3()]