
import numpy as np 

from GPy.models import GPRegression
from GPy.kern import RBF

from .. import moment_match
from .model import Model
from ..transform import BoxTransform, MeanTransform


class GPModel (Model):
	def __init__ (self, f, H, Q, R, delta_transition=True, transform=False):
		"""
		f : transition function x_{k+1} = f(x_k, u_k)
		H : observation matrix
		Q : process noise covariance matrix
		R : measurement noise covariance

		Model:
			x_{k+1} = g( x_k, u_k )  +  w_k,   w_k ~ N(0, Q)
				y_k = H * x_k  +  v_k,         v_k ~ N(0, R)
		
		if delta_transition:
			g( x_k, u_k ) = x_k  +  f( x_k, u_k )
		else
			g( x_k, u_k ) = f( x_k, u_k )

		transform : transform [x,u] -> [0, 1]^dim, and ( x_{k+1} - m) / std
		## WARNING - transform suffering problems ##

		We put a GP prior on f
		"""
		super(GPModel, self).__init__(f, H, Q, R)

		self.gps = []
		self.hyp = []
		self.transform = transform
		if self.transform:
			self.z_transform  = None
			self.t_transform  = None
		self.delta_transition = delta_transition

	"""
	Train GP surrogates
	"""
	def train (self, X, U, Z=None, hyp=None, noise_var=1e-6):
		assert len(U) == len(X)
		
		if Z is None:
			if self.delta_transition:
				Z = np.array([ self.f(x,u) - x for x,u in zip(X,U) ])
			else:
				Z = np.array([ self.f(x,u) for x,u in zip(X,U) ])
		Tt, Zt = self.training_data(np.c_[ X, U ], Z)

		self.hyp = []
		for d in range(self.D):
			dim = Tt.shape[1]
			gp  = GPRegression(Tt, Zt[:,[d]], RBF(input_dim=dim, ARD=True))
			if hyp is None:
				# Constrain noise variance
				gp.Gaussian_noise.variance.constrain_fixed(noise_var)
				# Constrain lengthscales
				LS = np.max(Tt, axis=0) - np.min(Tt, axis=0)
				for dd in range(dim):
					gp.kern.lengthscale[[dd]].constrain_bounded(
						lower=0., upper=10.*LS[dd], warning=False )
				gp.optimize()
			else:
				gp.update_model(False)
				gp.initialize_parameter()
				gp[:] = hyp[d]
				gp.update_model(True)
			self.hyp.append( gp[:] )
			self.gps.append( gp )

	"""
	Transform training data
	"""
	def training_data (self, T, Z):
		if not self.transform:
			return T, Z
		self.z_transform = MeanTransform( Z )
		self.t_transform = BoxTransform( T )
		return self.t_transform(T), self.z_transform(Z)

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Pk, u, cross_cov=False):
		if self.transform:
			assert self.z_transform is not None
			assert self.t_transform is not None
			assert not self.gps == []

		xku  = np.array( xk.tolist() + u.tolist() )
		tnew = xku.copy()
		dim  = len( tnew )
		Pnew = np.zeros((dim, dim))
		Pnew[:self.D, :self.D] = Pk
		if self.transform:
			tnew = self.t_transform(tnew)
			Pnew = self.t_transform.cov(Pnew)

		M, S, V = self.moment_match(tnew, Pnew)
				
		if self.transform:
			M  = self.z_transform(M, back=True)
			S  = self.z_transform.cov(S, back=True)
			V *= self.t_transform.q[:,None] * self.z_transform.q[None,:]
		V = V[:self.D]

		mu_k = M
		S_k  = S + self.Q
		if self.delta_transition:
			mu_k += xk
			S_k  += Pk + V + V.T
			V    += Pk
		return (mu_k, S_k, V) if cross_cov else (mu_k, S_k)


	def moment_match (self, mu, s2, grad=False):
		"""
		Inputs
			mu    input mean [ D ]
			s2    input covariance [ D, D ]
		"""
		assert not self.gps == []
		# Memory allocation
		D    = len( mu )             # Number of input dimensions
		E    = len( self.gps )       # Number of target dimensions
		N    = len( self.gps[0].X )  # Number of training data points

		logk = np.zeros( (N, E) )
		M    = np.zeros( (E) )     # Marginal mean, E_{x,f}[ f(x) ]
		S    = np.zeros( (E, E) )  # Marginal covariance, V_{x,f}[ f(x) ]
		V    = np.zeros( (D, E) )  # Input-output covariance, cov[ x, f(x) ]

		if grad:
			dMdm = np.zeros( (E, D) )       # output mean by input mean
			dMds = np.zeros( (E, D, D) )    # output mean by input covariance
			dSdm = np.zeros( (E, E, D) )    # output covariance by input mean
			dSds = np.zeros( (E, E, D, D) ) # output covariance by input covar
			dVdm = np.zeros( (D, E, D) )    # output covariance by input mean
			dVds = np.zeros( (D, E, D, D) ) # output covariance by input covar


		# Centralise training inputs
		inp = self.gps[0].X - mu[None,:]

		def maha (a, b, Q):
			aQ, bQ = np.matmul(a,Q), np.matmul(b,Q)
			asum = np.expand_dims(np.sum(aQ*a, axis=1), 1)
			bsum = np.expand_dims(np.sum(bQ*b, axis=1), 0)
			ab   = np.matmul(aQ, b.T)
			return (asum+bsum) - 2*ab

		# 1) Compute predicted mean and input-output covariance
		for i in range(E):
			# First, some useful intermediate terms
			beta  = self.gps[i].posterior.woodbury_vector
			lengp = np.array(self.gps[i].kern.lengthscale)
			rho2  = np.array(self.gps[i].kern.variance)[0]
			logk[:,i] = np.log(rho2) - 0.5 * np.sum( (inp/lengp)**2, axis=1 )
			
			iL = np.diag(1. / np.array(self.gps[i].kern.lengthscale))
			nn = np.matmul( inp, iL )
			B  = np.matmul( iL, np.matmul(s2, iL) ) + np.eye(D)
			iQ = np.linalg.inv( B )
			iB = np.matmul( iL, np.matmul(iQ, iL) )
			xB = np.matmul(inp, iB)
			t  = np.matmul( nn, iQ )
			l  = np.exp( -0.5 * np.sum(nn * t, axis=1) )
			lb = l * beta[:,0]
			c  = rho2 / np.sqrt(np.linalg.det(B))
			M[i] = c * np.sum( lb )
			Vi = c * np.sum( lb[:,None] * xB, axis=0 )
			V[:,i] = np.matmul(s2, Vi )
			if grad:
				tL  = np.matmul( t, iL )
				tlb = tL * lb[:,None]
				dMdm[i] = c * np.matmul(tL.T, lb)
				dMds[i] = 0.5 * ( c * np.matmul(tL.T, tlb) - iB*M[i])
				dVdm[:,i,:] = 2 * np.matmul(s2, dMds[i])
				for d in range(D):
					dVids = 0.5 * (c*np.matmul((xB*xB[:,[d]]).T, tlb) - iB*Vi[d]
								- Vi[:,None]*iB[[d],:] - iB[:,[d]]*Vi[None,:] )
					for j in range(D):
						dVds[j,i] += s2[j,d] * dVids
					dVds[d,i,d] += Vi
			
		# 2) predictive covariance matrix
		# 2a) non-central moments
		for i in range(E):
			lengi = 1. / np.array(self.gps[i].kern.lengthscale)**2
			ii    = inp * lengi
			betai = self.gps[i].posterior.woodbury_vector

			#for j in range(i+1):
			for j in range(i,E):
				lengj = 1. / np.array(self.gps[j].kern.lengthscale)**2
				ij    = inp * lengj
				betaj = self.gps[j].posterior.woodbury_vector

				R    = np.matmul(s2, np.diag(lengi + lengj)) + np.eye(D)
				
				detR = np.linalg.det(R)
				iR   = np.linalg.inv(R)
				iRS  = np.matmul(iR, s2)
				Q1   = logk[:,[i]] + logk[:,[j]].T
				Q2   = maha(ii, -ij, 0.5*iRS)
				Q    = np.exp(Q1 + Q2)
				isdR = 1. / np.sqrt( detR )

				A = betai * betaj.T
				if i == 0 and j == 1:
					added_return = [R, iRS, Q1, ii, ij, Q2, Q, isdR, A.copy()]
				if i == j:
					# Incorporate model uncertainty
					A -= self.gps[i].posterior.woodbury_inv
				A = A * Q
				
				S[i,j] = np.sum( A ) * isdR
				S[j,i] = S[i,j]
				
				if grad:
					zi  = np.matmul(ii, iR)
					zj  = np.matmul(ij, iR)
					r,T = np.zeros(D), np.zeros((D,D))
					for d in range(D):
						B    = (zi[:,[d]] + zj[:,[d]].T ) * A
						r[d] = np.sum(B) * isdR
						T[d, :d+1] = np.sum(np.matmul(zi[:,:d+1].T, B), axis=1)\
									+ np.sum(np.matmul(B, zj[:,:d+1]), axis=0)
						T[:d+1, d] = T[d,:d+1]
					
					r -= M[i]*dMdm[j] + M[j]*dMdm[i] 
					dSdm[i,j], dSdm[j,i] = r, r
					T  = 0.5 * (isdR * T - S[i,j] * iR*(lengi + lengj)[:,None])
					T -= M[i]*dMds[j] + M[j]*dMds[i] 
					dSds[i,j], dSds[j,i] = T, T
					
			S[i,i] += np.array(self.gps[i].kern.variance)[0]
		# 2b) centralise moments
		S -= M[:,None] * M[None,:]

		if grad:
			return M, S, V, dMdm, dMds, dSdm, dSds, dVdm, dVds
		return M, S, V

