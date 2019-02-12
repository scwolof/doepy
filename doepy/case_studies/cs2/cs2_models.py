
import numpy as np 

from ...models import GPModel
from ...constraints import Box

class Model (GPModel):
	def __init__ (self, F, B, H, Q, R, name):
		f   = lambda x,u: np.matmul(F,x) + np.matmul(B,u)
		X,Y = self.training_grid(f)
		super(Model, self).__init__(f, X, Y, H, Q, R, name=name, delta_transition=True)

		self.X_constraints = [ Box(-0.25, 0.25), Box(0, 1) ]

		self.x0 = np.array([0, 1])
		self.P0 = 2 * self.Q.copy()

	def training_grid (self, f):
		x1 = np.linspace(-0.25, 0.25, 6)
		x2 = np.linspace(    0,    1, 6)
		u  = np.linspace(    0,  0.5, 6)
		x1, x2, u = np.meshgrid( x1, x2, u)
		Xt = np.c_[x1.flatten(), x2.flatten()]
		Ut = u.flatten()[:,None]
		Y  = np.array([ f(x,u) for x, u in zip(Xt,Ut) ])
		return np.c_[Xt, Ut], Y

class M1 (Model):
	def __init__ (self):
		F = np.array([[0.7, -0.15],[0.2, 0.95]])
		B = np.array([[0.06], [0.006]])
		H = np.array([[0, 1.]])
		Q = 5e-5*np.eye(2)
		R = np.array([[1e-4]])
		Model.__init__(self, F, B, H, Q, R, 'M1')
	

class M2 (Model):
	def __init__ (self):
		F = np.array([[0.7, -0.1],[0.17, 0.99]])
		B = np.array([[0.06], [0.006]])
		H = np.array([[0, 1.]])
		Q = 5e-5*np.eye(2)
		R = np.array([[1e-4]])
		Model.__init__(self, F, B, H, Q, R, 'M2')
	

class M3 (Model):
	def __init__ (self):
		F = np.array([[0.72, -0.12],[0.17, 0.95]])
		B = np.array([[0.06], [0.006]])
		H = np.array([[0, 1.]])
		Q = 5e-5*np.eye(2)
		R = np.array([[1e-4]])
		Model.__init__(self, F, B, H, Q, R, 'M3')
	

def get ():
	return [M1(), M2(), M3()]



