
import numpy as np 

from .design_criteria import HR

class ProblemInstance:
	def __init__ (self, models, num_steps, u_bounds, u_delta_lim, y_bounds, 
			y_constraint_prob=0.99, div_criterion=None):
		self.models     = models
		self.num_models = len(models)
		self.num_steps  = num_steps
		self.divergence = div_criterion or HR()

		self.Du = len(u_bounds)
		self.U = np.array([np.mean(u_bounds, axis=1)]*num_steps)
		self.u_delta  = u_delta_lim
		self.y_bounds = y_bounds
		self.y_const_prob = y_constraint_prob
		assert isinstance(self.y_const_prob, float) and 0<self.y_const_prob<1

	def objective (self):
		raise NotImplementedError

	def constraints (self):
		raise NotImplementedError

	def _build_problem (self):
		# Hard inequality constraints
		ineq = self._build_u_delta_constraints()
		# Divergence term
		D  = 0.
		# Means and variances of predicted states and observations 
		mX = [ model.x0.astype(np.float) for model in self.models ]
		sX = [ model.P0.astype(np.float) for model in self.models ]
		mY = [ None for model in self.models ]
		sY = [ None for model in self.models ]
		# Iterate over the control sequence
		for n in range( self.num_steps ):
			# Next control input
			u = self.U[n]
			# State prediction
			for i, model in enumerate( self.models ):
				mX[i],sX[i] = model._build_predict_x_dist(mX[i],sX[i], u)[:2]
				mY[i],sY[i] = model._build_predict_y_dist(mX[i],sX[i])
				#ineq.append( self._build_state_constraint(mY[i],sY[i]) )
			D += self.divergence( mY, sY )
		return -D, ineq

	def _build_u_delta_constraints (self):
		ineq = []
		# U delta constraints
		for n in range( 1, self.num_steps ):
			for j in range( self.Du ):
				#d_const1 = self.U[j][n] - self.U[j][n-1] + self.u_delta[j]
				#d_const2 = self.U[j][n-1] - self.U[j][n] + self.u_delta[j]
				# | U[n] - U[n-1] | <= u_delta
				"""
				d_const1 = self.U[j][n] - self.U[j][n-1] - self.u_delta[j]
				d_const2 = self.U[j][n-1] - self.U[j][n] - self.u_delta[j]
				ineq.append(d_const1)
				ineq.append(d_const2)
				"""
				d_const = -self.U[j][n]
				ineq.append(d_const)
		return ineq

	"""
	def _build_state_constraint (self, mY, mS):
		dist  = mvn(mY, mS)
		Phi_b = dist.cdf(self.y_bounds[:,1])
		Phi_a = dist.cdf(self.y_bounds[:,0])
		return Phi_b - Phi_a - self.y_const_prob
	"""