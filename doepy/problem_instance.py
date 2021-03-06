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

class ProblemInstance:
	def __init__ (self, models, num_steps, div_criterion, u_bounds,
				  u_constraints=[], z_constraints=[]):
		self.models      = models
		self.num_models  = len( models )
		self.num_meas    = models[0].num_meas
		self.num_steps   = num_steps
		self.divergence  = div_criterion
		self.u_bounds    = u_bounds.copy()
		self.num_control = u_bounds.shape[0]

		# Control input bounds at every step
		self.bounds = np.array([ self.u_bounds ] * self.num_steps )
		self.bounds = self.bounds.reshape((self.num_steps*self.num_control, 2))

		# Lists of constraints
		self.u_constraints = u_constraints
		if not isinstance(self.u_constraints, list):
			self.u_constraints = [ self.u_constraints ]
		self.z_constraints = z_constraints
		if not isinstance(self.z_constraints, list):
			self.z_constraints = [ self.z_constraints ]

		# Compute the number of constraints
		self.num_constraints = 0
		for const in self.u_constraints:
			# Control constraints
			self.num_constraints += const.num_constraints( self.num_steps )
		for const in self.z_constraints:
			# Observed state constraints
			self.num_constraints += self.num_steps * self.num_models \
									* const.num_constraints()
		for model in self.models:
			# Latent state constraints - for data-driven surrogate models
			self.num_constraints += self.num_steps * model.num_x_constraints()


	def sample_U (self, check_constraints=False):
		# U : sampled control signal
		# C : (boolean) checks if U satisfies control constraints
		ul, uu = self.u_bounds[:,0], self.u_bounds[:,1]
		U = ul + (uu - ul) * np.random.rand(self.num_steps, self.num_control)
		if not check_constraints:
			return U
		C = [ np.all( const(U) >= 0.) for const in self.u_constraints ]
		return U, np.all( C )


	def __call__ (self, u_flat):
		E = self.num_meas
		N = self.num_steps
		M = self.num_models
		D = self.num_control

		# Unflatten
		U = u_flat.reshape(( N, D ))

		# Objective
		f    = 0.
		dfdU = np.zeros( U.shape )
		# Constraints
		C    = np.zeros( self.num_constraints )
		dCdU = np.zeros((self.num_constraints,) + U.shape )

		# Constraint counter
		i_c = 0
		# Control constraints
		for const in self.u_constraints:
			c, dc = const( U, grad=True )
			L     = const.num_constraints(self.num_steps)
			C[ i_c: i_c+L ]    = c
			dCdU[ i_c: i_c+L ] = dc
			i_c += L

		# Initial states
		x, s, dxdU, dsdU = [], [], [], []
		for model in self.models:
			x.append( model.x0 )
			s.append( model.x0_covar )
			dxdU.append( np.zeros(( N, model.num_states, D)) )
			dsdU.append( np.zeros(( N, model.num_states, model.num_states, D)) )
			model.initialise_x_constraints()
		Z = np.zeros(( M, E ))
		S = np.zeros(( M, E, E ))
		dZdU = np.zeros(( M, E, N, D))
		dSdU = np.zeros(( M, E, E, N, D))

		def gradchain (do, dxdU, dsdU, i, j):
			dMdU = np.matmul( do.dMdx, dxdU[i][j] ) \
			            + np.einsum( 'ijk,jkn->in', do.dMds, dsdU[i][j] )
			dSdU = np.matmul( do.dSdx, dxdU[i][j] ) \
			            + np.einsum( 'imjk,jkn->imn', do.dSds, dsdU[i][j] )
			return dMdU, dSdU
			       

		# Iterate over control sequence
		for n, u in enumerate( U ):
			dZdU.fill(0.)
			dSdU.fill(0.)

			# Predictive distributions at time n for model i
			for i, model in enumerate( self.models ):
				x[i], s[i], dox = model.predict_x_dist(x[i], s[i], u, grad=True)
				Z[i], S[i], doy = model.predict_z_dist(x[i], s[i], grad=True)
				for j in range( n+1 ):
					dxdU[i][j], dsdU[i][j] = gradchain(dox, dxdU, dsdU, i, j)
					if j == n:
						dxdU[i][j] += dox.dMdu
						dsdU[i][j] += dox.dSdu
					dZdU[i,:,j], dSdU[i,:,:,j] = gradchain(doy, dxdU, dsdU, i, j)

				# Update latent state constraints
				model.update_x_constraints(x[i], s[i], dxdU[i], dsdU[i])

				# State constraint for model i at time n
				for const in self.z_constraints:
					c, dcdZ, dcdS = const(Z[i], S[i], step=n, grad=True)
					L = const.num_constraints()
					C[ i_c: i_c+L ]    = c
					dCdU[ i_c: i_c+L ] = np.einsum('ij,jnk->ink',dcdZ,dZdU[i]) \
					                   + np.einsum('ijk,jknd->ind',dcdS,dSdU[i])
					i_c += L

			# Divergence between predictive distributions at time n
			for i, model in enumerate( self.models ):
				# Add measurement noise covariance
				S[i] += model.y_covar
			ftmp, dDdY, dDdS = self.divergence(Z, S, grad=True)
			f -= ftmp   ## Minimisation -> negative maximisation
			for j in range( n+1 ):
				dfdU[j] -= np.einsum('ij,ijk->k', dDdY, dZdU[:,:,j] ) \
				         + np.einsum('ijk,ijkl->l', dDdS, dSdU[:,:,:,j])

		# latent state constraints
		for i, model in enumerate( self.models ):
			res = model.get_x_constraints()
			if not res is None:
				L = res[0].shape[0]
				C[ i_c: i_c+L ]    = res[0]
				dCdU[ i_c: i_c+L ] = res[1]
				i_c += L

		# flatten
		dfdU = dfdU.reshape(u_flat.shape)
		dCdU = dCdU.reshape((-1,) + u_flat.shape)
		return f, C, dfdU, dCdU
