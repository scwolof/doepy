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

class DesignCriterion:
	def __call__ (self, M, S, grad=False):
		"""
		Input:
		M	[num_models x num_meas]             Matrix of predictive means
		S	[num_models x num_meas x num_meas]  Matrix of predictive covariances
		grad                                    Return gradients wrt M and S

		Output:
		dc      Design criterion for M, S
		dcdM    Gradient d dc / d M
		dcdS    Gradient d dc / d S
		"""
		return self._criterion(M, S, grad=grad)


class NullCriterion (DesignCriterion):
	"""
	Constant zero objective function
	"""
	def _criterion (self, M, S, grad=False):
		if not grad:
			return 0.
		dcdM = np.zeros(M.shape)
		dcdS = np.zeros(M.shape + (M.shape[1],))
		return 0., dcdM, dcdS


class HR (DesignCriterion):
	"""
	Hunter and Reiner's design criterion

	- Hunter and Reiner (1965)
	    Designs for discriminating between two rival models.
	    Technometrics 7(3):307-323
	"""
	def __init__ (self, W=None):
		DesignCriterion.__init__(self)
		# Scale dimensions
		self.W = W 

	def _criterion (self, M, S, grad=False):
		num_models, num_meas = M.shape
		if grad:
			dcdM = np.zeros((num_models, num_meas))
			dcdS = np.zeros((num_models, num_meas, num_meas))

		if self.W is None:
			self.W = np.eye( M[0].shape[0] )

		dc = 0.
		for i in range( num_models - 1 ):
			for j in range( i+1, num_models ):
				m = M[i] - M[j]
				if self.W.ndim == 1:
					Wm = self.W * m
				else:
					Wm = np.matmul(self.W, m)
				dc += np.sum(m * Wm)
				if grad:
					dcdM[i] += 2*Wm
					dcdM[j] -= 2*Wm

		return dc if not grad else (dc, dcdM, dcdS)


class BH (DesignCriterion):
	"""
	Box and Hill's design criterion, extended to multiresponse 
	models by Prasad and Someswara Rao.

	- Box and Hill (1967)
	    Discrimination among mechanistic models.
	    Technometrics 9(1):57-71
	- Prasad and Someswara Rao (1977)
	    Use of expected likelihood in sequential model 
	    discrimination in multiresponse systems.
	    Chem. Eng. Sci. 32:1411-1418
	"""
	def __init__ (self, w=None):
		DesignCriterion.__init__(self)
		# Model probability weights
		self.w = w

	def _criterion (self, M, S, grad=False):
		num_models, num_meas = M.shape
		if grad:
			dcdM = np.zeros((num_models, num_meas))
			dcdS = np.zeros((num_models, num_meas, num_meas))

		if self.w is None:
			self.w = np.ones( num_models )

		iS = np.linalg.inv( S )
		dc = 0
		for i in range( num_models - 1 ):
			for j in range( i+1, num_models ):
				SSj = np.matmul(S[i], iS[j])
				SSi = np.matmul(S[j], iS[i])
				t1  = np.trace( SSj + SSi - 2 * np.eye( num_meas ) )
				m   = M[i] - M[j]
				Wm  = np.matmul(iS[i] + iS[j], m)
				t2  = np.sum(m * Wm)
				wij = self.w[i] * self.w[j]
				dc += wij * (t1 + t2)
				if grad:
					dcdM[i] += wij * Wm
					dcdM[j] -= wij * Wm
					mm = m[:,None]*m[None,:]
					dcdS[i] += wij * ( iS[j].T - np.matmul(iS[i], SSi).T\
								 - np.matmul( iS[i], np.matmul( mm, iS[i].T ) ))
					dcdS[j] += wij * ( iS[i].T - np.matmul(iS[j], SSj).T\
								 - np.matmul( iS[j], np.matmul( mm, iS[j].T ) ))

		return 0.5*dc if not grad else (0.5*dc, dcdM, 0.5*dcdS)


class BF (DesignCriterion):
	"""
	Buzzi-Ferraris et al.'s design criterion.

	- Buzzi-Ferraris and Forzatti (1983)
	    Sequential experimental design for model discrimination 
	    in the case of multiple responses.
	    Chem. Eng. Sci. 39(1):81-85
	- Buzzi-Ferraris et al. (1984)
	    Sequential experimental design for model discrimination 
	    in the case of multiple responses.
	    Chem. Eng. Sci. 39(1):81-85
	- Buzzi-Ferraris et al. (1990)
	    An improved version of sequential design criterion for 
	    discrimination among rival multiresponse models.
	    Chem. Eng. Sci. 45(2):477-481
	"""
	def __init__ (self, noise_var):
		DesignCriterion.__init__(self)
		self.noise_var = noise_var

	def _criterion (self, M, S, grad=False):
		num_models, num_meas = M.shape
		if grad:
			dcdM = np.zeros((num_models, num_meas))
			dcdS = np.zeros((num_models, num_meas, num_meas))

		dc = 0
		for i in range( num_models - 1 ):
			for j in range( i+1, num_models ):
				iS  = np.linalg.inv(S[i] + S[j])
				SSi = 2 * np.matmul(self.noise_var, iS)
				t1  = np.trace( SSi )
				m   = M[i] - M[j]
				Wm  = np.matmul(iS, m)
				t2  = np.sum(m * Wm )
				dc += t1 + t2
				if grad:
					dcdM[i] += 2 * Wm
					dcdM[j] -= 2 * Wm
					dS = np.matmul(SSi.T + np.matmul(iS.T, m[:,None]*m[None,:]), iS.T)
					dcdS[i] -= dS
					dcdS[j] -= dS
		return dc if not grad else (dc, dcdM, dcdS)


class AW (DesignCriterion):
	"""
	Modified Expected Akaike Weights Decision Criterion.

	- Michalik et al. (2010). 
	    Optimal experimental design for discriminating numerous 
	    model candidates: The AWDC criterion.
	    Ind. Eng. Chem. Res. 49:913-919
	"""
	def __init__ (self, w=None, num_param=None):
		DesignCriterion.__init__(self)
		self.w = w
		self.num_param = num_param

	def _criterion (self, M, S, grad=False):
		num_models, num_meas = M.shape
		if grad:
			dcdM = np.zeros((num_models, num_meas))
			dcdS = np.zeros((num_models, num_meas, num_meas))

		if self.w is None:
			self.w = np.ones( num_models )
		if self.num_param is None:
			self.num_param = np.zeros( num_models )

		iS = np.linalg.inv( S )
		dc = np.zeros(num_models)
		for i in range( num_models ):
			if grad:
				dcdMi = np.zeros((num_models, num_meas))
				dcdSi = np.zeros((num_models, num_meas, num_meas))
			for j in range( num_models ):
				if i == j:
					dc[i] += 1
					continue
				m   = M[i] - M[j]
				Wm  = np.matmul(iS[i], m)
				t1  = np.sum(m * Wm)
				exp = np.exp(-0.5 * t1 + self.num_param[i] - self.num_param[j])
				dc[i] += exp
				if grad:
					dcdMi[i] += exp * Wm
					dcdMi[j] -= exp * Wm
					mm = m[:,None] * m[None,:]
					dcdSi[i] -= exp * np.matmul(iS[i], np.matmul(mm, iS[i])).T
			if grad:
				wdc2  = self.w[i] / dc[i]**2
				dcdM += wdc2 * dcdMi
				dcdS += wdc2 * 0.5 * dcdSi
		dc = np.sum( self.w / dc )
		return dc if not grad else (dc, dcdM, dcdS)


class JR (DesignCriterion):
	"""
	Quadratic Jensen-Renyi divergence.

	- Olofsson et al. (2019)
	    GPdoemd: a Python package for design of experiments for model discrimination
	    arXiv pre-print 1810.02561 (https://arxiv.org/abs/1810.02561)
	"""
	def __init__ (self, w=None):
		DesignCriterion.__init__(self)
		self.w = w

	def _criterion (self, M, S, grad=False):
		num_models, num_meas = M.shape
		if grad:
			dcdM  = np.zeros((num_models, num_meas))
			dT2dM = np.zeros((num_models, num_meas))

		if self.w is None:
			self.w = np.ones( num_models )

		# Pre-compute
		iS  = np.linalg.inv( S )
		dS  = np.linalg.det( S )
		ldS = np.log( dS )

		""" Sum of entropies """
		T1 = np.sum( self.w * 0.5 * ( num_meas*np.log(4*np.pi) + ldS ) )
		if grad:
			dcdS = -0.5 * self.w[:,None,None] * iS

		""" Entropy of sum """
		# Diagonal elements: (i,i)
		T2 = self.w**2 / ( 2**(num_meas/2.) * np.sqrt(dS) ) 
		if grad:
			dT2dS = -0.5 * T2[:,None,None] * iS
		T2 = np.sum( T2 )
		
		# Off-diagonal elements: (i,j)
		for i in range( num_models ):
			# mu_i^T * inv(Si) * mu_i 

			iSmi   = np.matmul(iS[i], M[i])
			miiSmi = np.matmul(M[i], iSmi)
			if grad:
				mimi    = M[i][:,None] * M[i][None,:]

			for j in range( i+1, num_models ):
				# mu_j^T * inv(Sj) * mu_j
				iSmj   = np.matmul(iS[j], M[j])
				mjiSmj = np.matmul(M[j], iSmj)

				# inv( inv(Si) + inv(Sj) )
				iSiS  = iS[i] + iS[j]
				iiSiS = np.linalg.inv( iSiS )
				liSiS = np.log( np.linalg.det( iSiS ))

				# mu_ij^T * inv( inv(Si) + inv(Sj) ) * mu_ij
				mij   = iSmi + iSmj
				iiSSj = np.matmul(mij, np.matmul(iiSiS, mij))

				phi = miiSmi + mjiSmj - iiSSj + ldS[i] + ldS[j] + liSiS
				exp = 2 * self.w[i] * self.w[j] * np.exp( -0.5 * phi )
				T2 += exp
				if grad:
					mjmj    = M[j][:,None] * M[j][None,:]
					iiSSij  = np.matmul(iiSiS, mij)
					iiSiSSi = np.matmul( iiSiS, iS[i] )
					iiSiSSj = np.matmul( iiSiS, iS[j] )

					dphidmi = iSmi - np.matmul(iS[i], iiSSij)
					dphidmj = iSmj - np.matmul(iS[j], iiSSij)
					
					dphidsi = -np.matmul(iS[i], np.matmul(mimi, iS[i])) + iS[i]\
					    - np.sum(np.sum(iiSiS[None,None,:,:] * \
					    ( iS[i,:,None,:,None] * iS[i,None,:,None,:] ),axis=3),axis=2)
					dphidsj = -np.matmul(iS[j], np.matmul(mjmj, iS[j])) + iS[j]\
					    - np.sum(np.sum(iiSiS[None,None,:,:] * \
					    ( iS[j,:,None,:,None] * iS[j,None,:,None,:] ),axis=3),axis=2)

					mmij    = mij[:,None] * mij[None,:]
					for n in range(num_meas):
						for m in range(num_meas):
							Smmi = mij[:,None] * M[i]
							Smmj = mij[:,None] * M[j]
							dphidsi[n,m] += 2 * np.sum(iiSiS.T[:,None,:] \
							        * Smmi[:,:,None] \
							        * (iS[i][:,[m]]*iS[i][:,n])[None,:,:]) \
							        - np.sum(mmij * iiSiSSi[:,[n]] * iiSiSSi[:,m])
							dphidsj[n,m] += 2 * np.sum(iiSiS.T[:,None,:] \
							        * Smmj[:,:,None] \
							        * (iS[j][:,[m]]*iS[j][:,n])[None,:,:]) \
							        - np.sum(mmij * iiSiSSj[:,[n]] * iiSiSSj[:,m])
					
					dT2dM[i] -= exp * dphidmi
					dT2dM[j] -= exp * dphidmj
					dT2dS[i] -= exp * 0.5 * dphidsi
					dT2dS[j] -= exp * 0.5 * dphidsj


		dc = 0.5 * num_meas * np.log(2*np.pi) - np.log( T2 ) - T1
		if grad:
			dcdM -= 1. / T2 * dT2dM
			dcdS -= 1. / T2 * dT2dS
		return dc if not grad else (dc, dcdM, dcdS)