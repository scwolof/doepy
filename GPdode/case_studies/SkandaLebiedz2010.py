
import numpy as np 
from scipy.integrate import odeint

"""
D. Skanda and D. Lebiedz (2010)
"An optimal experimental design approach to model discrimination
in dynamic biochemical systems"
Bioinformatics 26(7): 939-945
"""

class Model:
	def __init__ (self):
		self.num_states   = 2
		#self.num_controls = 2
		self.num_measured = 2

		self.H = np.eye(2)        # Observation matrix
		self.Q = np.zeros((2,2))  # Process noise covariance
		self.R = np.eye(2)        # Measurement noise covariance

		self.x0 = np.array([ 14.7, 2.21 ])

		#self.T  = 72.0
		#self.dt = 0.75
		#self.num_steps = 97

		#self.u_bounds = np.array([[0.05, 0.2], [5., 35.]])

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
		#t = np.linspace(0, self.dt, 26)
		X = odeint(f, x, t)
		return X[-1]


class M1 (Model):
	"""
	Allosteric model with positive feedback and linear product sink
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 0.879, 2.03, 0.109, 1.61e4 ])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		x1, x2 = x
		#u1, u2 = u
		p1, p2, p3, p4 = p 

		r   = ( x1 * (1+x1) * (1+x2)**2 ) / ( p4 + (1+x1)**2 * (1+x2)**2 )
		dx1 = 0.22 - p1 * r
		dx2 = (p2 * p1 * r) - (p3 * x2)
		return np.array([dx1, dx2])


class M2 (Model):
	"""
	Allosteric model with positive feedback in the absence of cooperativity
	when the product sink is represented by Michaelis-Menten kinetics
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 6.47, 3.33, 5.11, 244 ])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		x1, x2 = x
		#u1, u2 = u
		p1, p2, p3, p4 = p 

		r   = ( x1 * (1+x2) ) / ( p4 + (1+x1) * (1+x2) )
		dx1 = 0.22 - r
		dx2 = p1 * r - (p2 * x2) / (p3 + x2)
		return np.array([dx1, dx2])


class DataGen (M1):
	def __init__ (self):
		M1.__init__(self)
		self.true_param = np.array([ 0.889, 2.16, 0.111, 1.76e4 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2()]