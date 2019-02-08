
import numpy as np 

from GPy.models import GPRegression
from GPy.kern import RBF

from .model import Model
from ..transform import BoxTransform, MeanTransform

from pdb import set_trace as st

class GPModel (Model):
	def __init__ (self, f, H, Q, R, num_inputs, delta_transition=False, transform=True):
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
		super(GPModel, self).__init__(f, H, Q, R, num_inputs)

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
		Tt, Zt = self._training_data(np.c_[ X, U ], Z)

		self.hyp = []
		for d in range(self.num_states):
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
	def _training_data (self, T, Z):
		if not self.transform:
			return T, Z
		self.z_transform = MeanTransform( Z )
		self.t_transform = BoxTransform( T )
		return self.t_transform(T), self.z_transform(Z)

	"""
	State prediction
	"""
	def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False):
		if self.transform:
			assert self.z_transform is not None
			assert self.t_transform is not None
		assert not self.gps == [], 'GP surrogate(s) not trained yet.'

		# Input mean and variance
		tnew = np.array( xk.tolist() + u.tolist() )
		dim  = len( tnew )
		Snew = np.zeros((dim, dim))
		Snew[:self.num_states, :self.num_states] = Sk
		if self.transform:
			tnew = self.t_transform(tnew)
			Snew = self.t_transform.cov(Snew)

		# Moment matching
		res = self._moment_match(tnew, Snew, grad=grad)
		if grad:
			M, S, V, dMdt, dMds, dSdt, dSds, dVdt, dVds = res
		else:
			M, S, V = res
				
		# Transform back
		if self.transform:
			qt,qz = self.t_transform.q, self.z_transform.q
			M     = self.z_transform(M, back=True)
			S     = self.z_transform.cov(S, back=True)
			qtqz  = qt[:,None] * qz[None,:]
			V    *= qtqz
			if grad:
				qtqt  = qt[:,None] * qt[None,:]
				qzqz  = qz[:,None] * qz[None,:]
				dMdt *= qz[:,None] / qt[None,:]
				dMds *= qz[:,None,None] / qtqt[None,:,:]
				dSdt *= qzqz[:,:,None] / qt[None,None,:]
				dSds *= qzqz[:,:,None,None] / qtqt[None,None,:,:]
				dVdt *= qtqz[:,:,None] / qt[None,None,:]
				dVds *= qtqz[:,:,None,None] / qtqt[None,None,:,:]

		# Separate state and control dimensions again
		V = V[:self.num_states]
		if grad:
			dMdx = dMdt[:,:self.num_states]
			dMdu = dMdt[:,self.num_states:]
			dMds = dMds[:,:self.num_states,:self.num_states]
			dSdx = dSdt[:,:,:self.num_states]
			dSdu = dSdt[:,:,self.num_states:]
			dSds = dSds[:,:,:self.num_states,:self.num_states]
			dVdx = dVdt[:self.num_states,:,:self.num_states]
			dVdu = dVdt[:self.num_states,:,self.num_states:]
			dVds = dVds[:self.num_states,:,:self.num_states,:self.num_states]

		# Process noise variance
		S += self.Q
		# Delta transition
		if self.delta_transition:
			M += xk
			S += Sk + V + V.T
			V += Sk
			if grad:
				dMdx += np.eye(self.num_states)
				dSdx += dVdx + np.swapaxes(dVdx,0,1)
				dSds += dVds + np.swapaxes(dVds,0,1)
				for d1 in range(self.num_states):
					for d2 in range(self.num_states):
						dSds[d1,d2,d1,d2] += 1
						dVds[d1,d2,d1,d2] += 1
		# Returns
		if not grad:
			return (M, S, V) if cross_cov else (M, S)
		if not cross_cov:
			return M, S, dMdx, dMds, dMdu, dSdx, dSds, dSdu
		return M, S, V, dMdx, dMds, dMdu, dSdx, dSds, dSdu, dVdx, dVds, dVdu


	def _moment_match (self, mu, s2, grad=False):
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

		for e in range( E ):
			# First, some useful intermediate terms
			beta = self.gps[e].posterior.woodbury_vector.flatten()
			leng = np.array(self.gps[e].kern.lengthscale)
			rho2 = np.array(self.gps[e].kern.variance)[0]
			
			iL    = np.diag(1. / leng**2 )
			is2L  = np.linalg.inv( s2 + np.diag(leng**2) )
			is2LX = np.matmul(inp, is2L)
			q     = np.exp(-0.5 * np.sum(inp * is2LX, axis=1))
			bq    = beta * q
			s2LI  = np.matmul(s2, iL) + np.eye(D)
			c     = rho2 / np.sqrt(np.linalg.det( s2LI ))
			
			sumbq = np.sum( bq )
			M[e]  = c * sumbq
			if grad:
				dbqdm   = bq[:,None] * is2LX
				dbqds   = 0.5 * bq[:,None,None]*is2LX[:,:,None]*is2LX[:,None,:]
				dcds    = -0.5 * np.matmul( np.linalg.inv(s2LI).T, iL )
				dMdm[e] = c * np.sum( dbqdm, axis=0 )
				dMds[e] = c * ( np.sum( dbqds, axis=0 ) + dcds * sumbq )

			is2LXs2   = np.matmul(is2LX, s2.T)
			sumbqSS   = np.sum( bq[:,None] * is2LXs2, axis=0 )
			V[:,e]    = c * np.sum( bq[:,None] * is2LXs2, axis=0 )
			if grad:
				dVdm[:,e] = 2 * np.matmul(s2, dMds[e])
				dVds[:,e] =  dcds * V[:,e,None,None]
				s2is2L = np.matmul(s2, is2L)
				for d1 in range(D):
					dis2LXs2ds = - is2LX[:,None,d1,None] * s2is2L[None,:]
					dis2LXs2ds[:,d1] += is2LX
					dsumbqSSds = np.sum( dbqds[:,None,d1] * is2LXs2[:,:,None] \
									+ bq[:,None,None] * dis2LXs2ds, axis=0 )
					dVds[:,e,d1] += c * dsumbqSSds
				
			logk[:,e] = np.log(rho2) - 0.5 * np.sum( (inp/leng)**2, axis=1 )
			
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
				
				R     = np.matmul(s2, np.diag(lengi + lengj)) + np.eye(D)
				iR    = np.linalg.inv( R )
				isdR  = 1. / np.sqrt( np.linalg.det(R) )
				iRs2  = np.matmul( iR, s2 )
				zi,zj = np.matmul(ii,iRs2), np.matmul(ij,iRs2)
				i1,j2 = np.sum(zi*ii,1), np.sum(zj*ij,1)
				zRSz  = (i1[:,None] + j2[None,:]) + 2*np.matmul(zi,ij.T)
				Q     = np.exp( (logk[:,[i]] + logk[:,j]) + 0.5*zRSz )

				A = betai * betaj.T
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
					dSdm[i,j], dSdm[j,i] = r.copy(), r.copy()
					T  = 0.5 * (isdR * T - S[i,j] * iR*(lengi + lengj)[:,None])
					T -= M[i]*dMds[j] + M[j]*dMds[i] 
					dSds[i,j], dSds[j,i] = T.copy(), T.copy()
					
			S[i,i] += np.array(self.gps[i].kern.variance)[0]
		# 2b) centralise moments
		S -= M[:,None] * M[None,:]

		if grad:
			return M, S, V, dMdm, dMds, dSdm, dSds, dVdm, dVds
		return M, S, V
