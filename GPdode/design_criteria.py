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

import tensorflow as tf
from gpflow import settings

dtype = settings.float_type

class DesignCriterion:
	def __call__ (self, M, S):
		return self._criterion(M, S)

class HR (DesignCriterion):
	"""
	Hunter and Reiner's design criterion

	- Hunter and Reiner (1965)
		Designs for discriminating between two rival models.
		Technometrics 7(3):307-323
	"""
	def __init__ (self, W=None):
		DesignCriterion.__init__(self)
		self.W = W

	def _criterion (self, M, S):
		num_models = len( M )
		if self.W is None:
			self.W = tf.eye(tf.shape(M[0])[0], dtype=dtype)

		dc = 0.
		for i in range( num_models - 1 ):
			for j in range( i+1, num_models ):
				m   = ( M[i] - M[j] )[:,None]
				dc += tf.matmul(m, tf.matmul(self.W, m), transpose_a=True)
		return dc


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
		self.w = w

	def _criterion (self, M, S):
		num_models = len( M )
		if self.w is None:
			self.w = tf.constant([1]*num_models, dtype=dtype)
		E = tf.shape(M[0])[0]

		iS = [tf.linalg.inv(s2) for s2 in S]
		dc = 0
		for i in range( num_models - 1 ):
			for j in range( i+1, num_models ):
				t1  = tf.trace( tf.matmul(S[i], iS[j]) + tf.matmul(S[j], iS[i])\
							 - 2 * tf.eye( E, dtype=dtype ) )
				m   = ( M[i] - M[j] )[:,None]
				t2  = tf.matmul(m, tf.matmul(iS[i] + iS[j], m), transpose_a=True)
				dc += self.w[i] * self.w[j] * (t1 + t2)
		return 0.5 * dc


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
		self.noise_var = tf.constant(noise_var, dtype=dtype)

	def _criterion (self, M, S):
		num_models = len( M )

		dc = 0
		for i in range( num_models - 1 ):
			for j in range( i+1, num_models ):
				iS  = tf.linalg.inv(S[i] + S[j])
				t1  = tf.trace( tf.matmul(self.noise_var, iS) )
				m   = ( M[i] - M[j] )[:,None]
				t2  = tf.matmul(m, tf.matmul(iS, m), transpose_a=True)
				dc += t1 + t2
		return dc


class AW (DesignCriterion):
	"""
	Modified Expected Akaike Weights Decision Criterion.

	- Michalik et al. (2010). 
		Optimal Experimental Design for Discriminating Numerous 
		Model Candidates: The AWDC Criterion.
		Ind. Eng. Chem. Res. 49:913-919
	"""
	def __init__ (self, w=None, num_param=None):
		DesignCriterion.__init__(self)
		self.w = w
		self.num_param = num_param

	def _criterion (self, M, S):
		num_models = len( M )
		if self.w is None:
			self.w = tf.constant([1]*num_models, dtype=dtype)
		if self.num_param is None:
			self.num_param = tf.constant([0]*num_models, dtype=dtype)

		iS = [tf.linalg.inv(s2) for s2 in S]
		dc = 0.
		for i in range( num_models ):
			t2 = 0.
			for j in range( num_models ):
				m   = ( M[i] - M[j] )[:,None]
				t1  = tf.matmul( m, tf.matmul(iS[i], m), transpose_a=True )
				t2 += tf.exp( -0.5 * t1 + self.num_param[i] - self.num_param[j] )
			dc += self.w[i] / t2
		return dc

class JR (DesignCriterion):
	"""
	Quadratic Jensen-Renyi divergence.

	- Olofsson et al. (Future publication)
	"""
	def __init__ (self, w=None):
		DesignCriterion.__init__(self)
		self.w = w

	def _criterion (self, M, S):
		num_models = len( M )
		if self.w is None:
			self.w = tf.constant([1]*num_models, dtype=dtype)
		E = 0.5 * tf.shape( M[0], out_type=dtype )[0]

		# Pre-compute
		iS  = [tf.linalg.inv( s2 ) for s2 in S]
		dS  = [tf.linalg.det( s2 ) for s2 in S]
		ldS = [tf.log( ds2 ) for ds2 in dS]
		log2pi = tf.constant( 1.8378770664093453, dtype=dtype )

		""" Sum of entropies """
		T1 = 0.
		pw = E * (0.69314718056 + log2pi)
		for i in range( num_models ):
			T1 += self.w[i] * ( pw + 0.5 * ldS[i] )

		""" Entropy of sum """
		# Diagonal elements: (i,i)
		T2 = 0.
		pw = tf.pow( tf.constant(2.,dtype=dtype), E)
		for i in range( num_models ):
			T2 += self.w[i]**2 / ( pw * tf.sqrt(dS[i]) )
		
		# Off-diagonal elements: (i,j)
		for i in range( num_models ):
			# mu_i^T * inv(Si) * mu_i 
			mi     = M[i][:,None]
			iSmi   = tf.matmul(iS[i], mi)
			miiSmi = tf.matmul(mi, iSmi, transpose_a=True)

			for j in range( i+1, num_models ):
				# mu_j^T * inv(Sj) * mu_j
				mj     = M[j][:,None]
				iSmj   = tf.matmul(iS[j], mj)
				mjiSmj = tf.matmul(mj, iSmj, transpose_a=True)

				# inv( inv(Si) + inv(Sj) )
				iSiS  = iS[i] + iS[j]
				iiSiS = tf.linalg.inv( iSiS )
				liSiS = tf.log( tf.linalg.det( iSiS ))

				# mu_ij^T * inv( inv(Si) + inv(Sj) ) * mu_ij
				mij   = iSmi + iSmj
				iiSSj = tf.matmul(mij, tf.matmul(iiSiS, mij), transpose_a=True)

				phi = miiSmi + mjiSmj - iiSSj + ldS[i] + ldS[j] + liSiS
				T2 += 2 * self.w[i] * self.w[j] * tf.exp( -0.5 * phi )

		T2 = E * log2pi - tf.log( T2 )
		return T2 - T1



