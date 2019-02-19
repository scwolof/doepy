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
				  u_constraints=[], y_constraints=[]):
		self.models      = models
		self.num_models  = len( models )
		self.num_meas    = models[0].num_meas
		self.num_steps   = num_steps
		self.divergence  = div_criterion
		self.u_bounds    = u_bounds.copy()
		self.num_control = u_bounds.shape[0]

		# Lists of constraints
		self.u_constraints = u_constraints
		if not isinstance(self.u_constraints, list):
			self.u_constraints = [ self.u_constraints ]
		self.y_constraints = y_constraints
		if not isinstance(self.y_constraints, list):
			self.y_constraints = [ self.y_constraints ]

		# Compute the number of constraints
		self.num_constraints = 0
		for const in self.u_constraints:
			self.num_constraints += const.num_constraints( self.num_steps )
		for const in self.y_constraints:
			self.num_constraints += self.num_steps * self.num_models \
									* const.num_constraints()


	def sample_U (self):
		ul, uu = self.u_bounds[:,0], self.u_bounds[:,1]
		return ul + (uu - ul) * np.random.rand(num_steps, self.num_control)


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
		x, p, dxdU, dpdU = [], [], [], []
		for model in self.models:
			x.append( model.x0 )
			p.append( model.P0 )
			dxdU.append( np.zeros(( N, model.num_states, D)) )
			dpdU.append( np.zeros(( N, model.num_states, model.num_states, D)) )
		Y = np.zeros(( M, E ))
		S = np.zeros(( M, E, E ))
		dYdU = np.zeros(( M, E, N, D))
		dSdU = np.zeros(( M, E, E, N, D))

		# Iterate over control sequence
		for n, u in enumerate( U ):
			dYdU.fill(0.)
			dSdU.fill(0.)

			# Predictive distributions at time n for model i
			for i, model in enumerate( self.models ):
				x[i], p[i], dxdx, dxdp, dxdu, dpdx, dpdp, dpdu \
					= model.predict_x_dist(x[i], p[i], u, grad=True)
				Y[i], S[i], dydx, dydp, dsdx, dsdp \
					= model.predict_y_dist(x[i], p[i], grad=True)
				for j in range( n+1 ):
					dxdU[i][j], dpdU[i][j] \
						= np.matmul( dxdx, dxdU[i][j] ) \
								+ np.einsum( 'ijk,jkn->in', dxdp, dpdU[i][j] ),\
						  np.matmul( dpdx, dxdU[i][j] ) \
								+ np.einsum( 'imjk,jkn->imn', dpdp, dpdU[i][j] )
					if j == n:
						dxdU[i][j] += dxdu
						dpdU[i][j] += dpdu
					dYdU[i,:,j]   = np.matmul( dydx, dxdU[i][j] ) \
								+ np.einsum( 'ijk,jkn->in', dydp, dpdU[i][j] )
					dSdU[i,:,:,j] = np.matmul( dsdx, dxdU[i][j] ) \
								+ np.einsum( 'imjk,jkn->imn', dsdp, dpdU[i][j] )

				# State constraint for model i at time n
				for const in self.y_constraints:
					c, dcdY, dcdS = const(Y[i], S[i], grad=True)
					L = const.num_constraints()
					C[ i_c: i_c+L ]    = c
					dCdU[ i_c: i_c+L ] = np.einsum('ij,jnk->ink',dcdY,dYdU[i]) \
								+ np.einsum('ijk,jknd->ind',dcdS,dSdU[i])
					i_c += L

			# Divergence between predictive distributions at time n
			ftmp, dDdY, dDdS = self.divergence(Y, S, grad=True)
			f += ftmp
			for j in range( n+1 ):
				dfdU[j] += np.einsum('ij,ijk->k', dDdY, dYdU[:,:,j] ) \
							+ np.einsum('ijk,ijkl->l', dDdS, dSdU[:,:,:,j]) 

		# flatten
		dfdU = dfdU.reshape(u_flat.shape)
		dCdU = dCdU.reshape((self.num_constraints,) + u_flat.shape)
		return f, C, dfdU, dCdU
