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

class Model:
	def __init__ (self, num_states):
		self.num_states = num_states
		self.num_inputs = 2
		self.num_meas   = 2

		self.H = np.zeros((self.num_meas, self.num_states))
		self.H[0,0], self.H[1,2] = 1, 1

		self.R  = 2.5e-3 * np.eye( self.num_meas )
		self.Q  = 1e-4 * np.eye( self.num_states )
		self.Su = 1e-5 * np.eye( self.num_inputs )

		self.x0 = np.zeros( self.num_states )
		self.P0 = 1e-6 * np.eye( self.num_states )

		self.hessian = True

		self.num_steps = 50
		self.u_delta   = np.array([0.25] * self.num_inputs)
		self.u_bounds  = np.array([[-0.5, 0.5]] * self.num_inputs)
		self.y_bounds  = np.array([[-0.6, 0.6], [0., 1.]])

		# State space feasibility
		self.known_dim   = [0, 2]
		self.x_bounds    = np.array([[None, None]]*self.num_states)
		self.x_bounds[0] = self.y_bounds[0]
		self.x_bounds[2] = self.y_bounds[1]

	def get_candidate_dict (self):
		return {'f':  self,
		        'H':  self.H,
		        'Q':  self.Q,
		        'R':  self.R,
		        'x0': self.x0,
		        'S_u': self.Su,
		        'S_x0': self.P0,
		        'name': self.name,
		        'hessian': True,
		        'x_bounds': self.x_bounds,
		        'u_bounds': self.u_bounds,
		        'num_inputs': self.num_inputs}


class M1 (Model):
	def __init__ (self):
		super().__init__(3)
		self.name = 'M1'
		# State space feasibility
		self.x_bounds[1] = self.x_bounds[2]

	def __call__ (self, x, u, grad=False):
		x1, x2, x3 = x
		u1, u2     = u
		xk1 = u1
		xk2 = x3
		xk3 = x1 + x2 * u2
		g   = np.array([xk1, xk2, xk3])
		if not grad:
			return g
		# dgdx
		dgdx      = np.zeros((3,3))
		dgdx[1,2] = 1
		dgdx[2]   = np.array([1., u2, 0])
		# dgdu
		dgdu      = np.zeros((3,2))
		dgdu[0,0] = 1
		dgdu[2]   = np.array([0., x2])
		# ddgddx
		ddgddx = np.zeros((3,3,3))
		ddgddu = np.zeros((3,2,2))
		ddgdxu = np.zeros((3,3,2))
		ddgdxu[2,1,1] = 1
		return g, dgdx, dgdu, ddgddx, ddgddu, ddgdxu


class M2 (Model):
	def __init__ (self):
		super().__init__(3)
		self.name = 'M2'
		# State space feasibility
		self.x_bounds[1] = self.x_bounds[2] + self.x_bounds[0] - self.u_bounds[0][::-1]

	def __call__ (self, x, u, grad=False):
		x1, x2, x3 = x
		u1, u2     = u
		xk1 = u1
		xk2 = x3 + (x1 - u1)
		xk3 = x1 + x2 * u2
		g = np.array([xk1, xk2, xk3])
		if not grad:
			return g
		# dgdx
		dgdx    = np.zeros((3,3))
		dgdx[1] = np.array([1., 0., 1.])
		dgdx[2] = np.array([1., u2, 0])
		# dgdu
		dgdu = np.array([[1.,0],[-1,0.],[0.,x2]])
		# ddgddx
		ddgddx = np.zeros((3,3,3))
		ddgddu = np.zeros((3,2,2))
		ddgdxu = np.zeros((3,3,2))
		ddgdxu[2,1,1] = 1
		return g, dgdx, dgdu, ddgddx, ddgddu, ddgdxu


class M3 (Model):
	def __init__ (self):
		super().__init__(4)
		self.name = 'M3'
		# State space feasibility
		self.x_bounds[3] = self.x_bounds[0] - self.u_bounds[0][::-1]
		self.x_bounds[1] = self.x_bounds[2] + self.x_bounds[3]

	def __call__ (self, x, u, grad=False):
		x1, x2, x3, x4 = x
		u1, u2 = u
		xk1 = u1
		xk2 = x3 + x4
		xk3 = x1 + x2 * u2
		xk4 = x1 - u1
		g = np.array([xk1, xk2, xk3, xk4])
		if not grad:
			return g
		# dgdx
		dgdx    = np.zeros((4,4))
		dgdx[1] = np.array([0., 0., 1., 1.])
		dgdx[2] = np.array([1., u2, 0., 0.])
		dgdx[3] = np.array([1., 0., 0., 0.])
		# dgdu
		dgdu = np.array([[1.,0],[0.,0.],[0.,x2],[-1.,0.]])
		# ddgddx
		ddgddx = np.zeros((4,4,4))
		ddgddu = np.zeros((4,2,2))
		ddgdxu = np.zeros((4,4,2))
		ddgdxu[2,1,1] = 1
		return g, dgdx, dgdu, ddgddx, ddgddu, ddgdxu


class M4 (Model):
	def __init__ (self):
		super().__init__(4)
		self.name = 'M4'
		# State space feasibility
		self.x_bounds[3] = self.x_bounds[0] - self.u_bounds[0][::-1]
		tmp = self.x_bounds[0][:,None] * self.x_bounds[3][None,:]
		self.x_bounds[3] = np.array([ np.min(tmp), np.max(tmp) ])
		self.x_bounds[1] = self.x_bounds[2] + self.x_bounds[3]

	def __call__ (self, x, u, grad=False):
		x1, x2, x3, x4 = x
		u1, u2 = u
		xk1 = u1
		xk2 = x3 + x4
		xk3 = x1 + x2 * u2
		xk4 = x1 * (x1 - u1)
		g = np.array([xk1, xk2, xk3, xk4])
		if not grad:
			return g
		# dgdx
		dgdx    = np.zeros((4,4))
		dgdx[1] = np.array([0., 0., 1., 1.])
		dgdx[2] = np.array([1., u2, 0., 0.])
		dgdx[3] = np.array([2.*x1 - u1, 0., 0., 0.])
		# dgdu
		dgdu = np.array([[1.,0],[0.,0.],[0.,x2],[-x1,0.]])
		# ddgddx
		ddgddx = np.zeros((4,4,4))
		ddgddx[3,0,0] = 2.
		ddgddu = np.zeros((4,2,2))
		ddgdxu = np.zeros((4,4,2))
		ddgdxu[2,1,1] =  1.
		ddgdxu[3,0,0] = -1.
		
		return g, dgdx, dgdu, ddgddx, ddgddu, ddgdxu


class DataGen (M4):
	def __init__ (self):
		super().__init__()

def get ():
	return DataGen(), [M1(), M2(), M3(), M4()]