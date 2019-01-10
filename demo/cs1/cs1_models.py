
import numpy as np 

from ...models import LinearModel
from ...constraints import Box

class Model (LinearModel):
	def __init__ (self, F, B, H, Q, R, name):
		super(Model, self).__init__(F, B, H, Q, R, name=name, delta_transition=True)

		#self.X_constraints = [ Box(-0.25, 0.25), Box(0, 1) ]
		self.X_constraints = np.array([[-0.25, 0.25],
									   [    0,    1]])

		self.x0 = np.array([0,1])
		self.P0 = 2 * self.Q.copy()

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



