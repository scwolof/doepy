
import numpy as np 
from scipy.integrate import odeint

"""
A. Mesbah, S. Streif, R. Findeisen and R. D. Braatz (2014)
"Active fault diagnosis for nonlinear systems with probabilistic uncertainties"
IFAC Proceedings (2014): 7079-7084
"""

class Model:
	def __init__ (self):
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

	def __call__ (self, x, u, p):
		"""
		Inputs:
		   x    States at time t:   x[i] - water level tank i (m)
		   u    Controls at time t: u[i] - water inflow (m^3 / s)
		   p    Model parameters
		Outputs:
		   dx   States at time t+1
		"""
		f = lambda y,t: self._ode_func(y,u,p)
		t = np.linspace(0, self.dt, 51)
		X = odeint(f, x, t)
		return X[-1]

	def _t_dx (self, x, u, p):
		x1, x2, x3 = x
		p1, p2, p3 = p[:3]
		t1  = p1 * np.sign(x1-x3) * np.sqrt(2*self.g*np.abs(x1-x3))
		t2  = p2 * np.sqrt(2*self.g*x2)
		t3  = p3 * np.sign(x3-x2) * np.sqrt(2*self.g*np.abs(x3-x2))
		dx1 = -t1 * self.Sp
		dx2 = ( t3 - t2 ) * self.Sp
		dx3 = ( t1 - t3 ) * self.Sp
		return t1, t2, t3, dx1, dx2, dx3


class M1 (Model):
	"""
	Nominal scenario (no fault)
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 1., 0.8, 1. ])
		self.p_bounds  = np.array([[0.9, 1.1], [0.7, 0.9], [0.9, 1.1]])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		t1, t2, t3, dx1, dx2, dx3 = self._t_dx(x, p)
		return np.array([dx1 + u[0], dx2 + u[1], dx3]) / self.A


class M2 (Model):
	"""
	Multiplicative actuator fault in inlet pump
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 1., 0.8, 1., 0.6 ])
		self.p_bounds  = np.array([[0.9, 1.1],[0.7, 0.9],[0.9, 1.1],[0.5,0.7]])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		t1, t2, t3, dx1, dx2, dx3 = self._t_dx(x, p)
		return np.array([dx1 + u[0]*p[3], dx2 + u[1], dx3]) / self.A


class M3 (Model):
	"""
	Circular leak in tank
	"""
	def __init__ (self):
		Model.__init__(self)
		self.p0 = np.array([ 1., 0.8, 1., 0.002 ])
		self.p_bounds  = np.array([[0.9, 1.1],[0.7, 0.9],[0.9, 1.1],[0., 0.004]])
		self.num_param = len( self.p0 )

	def _ode_func (self, x, u, p):
		t1, t2, t3, dx1, dx2, dx3 = self._t_dx(x, p)
		leak = t2 * np.pi * p[3]**2
		return np.array([dx1 + u[0], dx2 - leak + u[1], dx3]) / self.A


class DataGen (M2):
	def __init__ (self):
		M1.__init__(self)
		self.true_param = np.array([ 0.97, 0.82, 0.96, 0.67 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2(), M3()]