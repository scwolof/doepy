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
from numpy.random import multivariate_normal as mvn
from scipy.integrate import solve_ivp

from ..derivatives import LatentStateDerivatives
from ..statespacemodel import StateSpaceModel
from ...utils import assert_is_shape

class ctModel (StateSpaceModel):
    def __init__ (self, candidate_model):
        """
        Model:
            d x / dt = f( x(t), u, p )  +  w(t),   w(t) ~ N(0, x_covar)
                 z_k = H * x_k
                 y_k = z_k  +  v_k,                v_k ~ N(0, y_covar)
        with 
            x_0 ~ N(x0, x0_covar), 
            u_k ~ N(u_k, u_covar), 
            p   ~ N(p, p_covar)
            u_k of dimension num_inputs
        """
        super().__init__(candidate_model)
        self.step_length = candidate_model.step_length
        self.time_steps  = candidate_model.time_steps
        if self.time_steps is None:
            self.time_steps = 101

    def _predict (self, x, u, T):
        ps = mvn(self.p_mean,self.p_covar)
        args = (u,) if self.num_param < 1 else (u, ps)
        ode  = lambda t, x: self.f(x,*args)
        #T    = self._get_time_steps(start=start_time)
        #T    = (T[0], T[-1])
        xk1  = solve_ivp(ode, T, x)['y'][:,-1]
        yk   = np.matmul(self.H, x)
        return xk1, yk

    def _sample (self, x, u, T):
        us = mvn(u, self.u_covar)
        xy = self.predict(x, us, T)
        #wk = mvn( np.zeros(self.num_states), self.step_length*self.x_covar )
        wk = mvn( np.zeros(self.num_states), self.x_covar )
        vk = mvn( np.zeros(self.num_meas), self.y_covar )
        #return xy[0] + wk, xy[1] + vk
        return xy[0], xy[1] + vk
    
    def _ode_vector_merge (self, x, S, do=None, grad=False):
        assert_is_shape(x, (self.num_states,))
        assert_is_shape(S, (self.num_states, self.num_states))
        
        # Mean, covariance and input-output covariance
        xs = ( x[:,None], S )
        
        if do is None:
            do = LatentStateDerivatives(self)
            do.dMdx = np.eye(self.num_states)
            if grad:
                do.dSds = np.eye(self.num_states)[:,None,:,None] \
                          * np.eye(self.num_states)[None,:,None,:]
        
        # dMdx, dMdu, dMdp
        xs += ( do.dMdx, do.dMdu )
        if self.num_param > 0:
            xs  += ( do.dMdp, )
        if not grad:
            return np.concatenate(xs, axis=1).T.flatten()
        
        # dMds, dSdx, dSdu, dSdp
        xs += ( do.dMds.reshape(( self.num_states, -1 )),
                do.dSdx.reshape(( self.num_states, -1 )),
                do.dSdu.reshape(( self.num_states, -1 )) )
        if self.num_param > 0:
            xs  += ( do.dSdp.reshape(( self.num_states, -1 )), )
        # dSds
        xs += ( do.dSds.reshape(( self.num_states, -1 )), )

        # dMdxx
        xs += ( do.dMdxx.reshape(( self.num_states, -1 )), )
        # dMdxu
        xs += ( do.dMdxu.reshape(( self.num_states, -1 )), )
        # dMduu
        xs += ( do.dMduu.reshape(( self.num_states, -1 )), )
        if self.num_param > 0:
            # dMdxp
            xs += ( do.dMdxp.reshape(( self.num_states, -1 )), )
            # dMdup
            xs += ( do.dMdup.reshape(( self.num_states, -1 )), )
            # dMdpp
            xs += ( do.dMdpp.reshape(( self.num_states, -1 )), )
        
        return np.concatenate(xs, axis=1).T.flatten()
    
    def _ode_vector_unmerge (self, X, grad=False):
        # Mean x
        d = [1]
        # Covariance S
        d += [ self.num_states ]
        
        # dMdx, dMdu, dMdp
        d += [self.num_states, self.num_inputs, self.num_param]

        if grad:
            # dMds
            d += [ self.num_states*self.num_states ]
            # dSdx, dSdu, dSdp
            d += [ self.num_states*self.num_states ]
            d += [ self.num_states*self.num_inputs ]
            d += [ self.num_states*self.num_param ]
            # dSds
            d += [ self.num_states*self.num_states*self.num_states ]
            # dMdxx, dMdxu, dMduu
            d += [ self.num_states*self.num_states ]
            d += [ self.num_states*self.num_inputs ]
            d += [ self.num_inputs*self.num_inputs ]
            if self.num_param > 0:
                # dMdxp, dMdup, dMdpp
                d += [ self.num_states*self.num_param ]
                d += [ self.num_inputs*self.num_param ]
                d += [ self.num_param*self.num_param ]
            
        D = np.cumsum( d )
        X = X.reshape((D[-1], self.num_states)).T
        
        x = X[:,0];
        i = 0
        S = X[:,D[i]:D[i+1]]; i+= 1
        
        do = LatentStateDerivatives(self)
        Dx = (self.num_states,)
        Du = (self.num_inputs,)
        Dp = (self.num_param,)
        
        do.dMdx = X[:,D[i]:D[i+1]]; i+= 1
        do.dMdu = X[:,D[i]:D[i+1]]; i+= 1
        if self.num_param > 0:
            do.dMdp = X[:,D[i]:D[i+1]]
        i+= 1
        if not grad:
            return x, S, do

        do.dMds = X[:,D[i]:D[i+1]].reshape(Dx*3); i+= 1
        do.dSdx = X[:,D[i]:D[i+1]].reshape(Dx*3); i+= 1
        do.dSdu = X[:,D[i]:D[i+1]].reshape(Dx*2+(self.num_inputs,)); i+= 1
        if self.num_param > 0:
            do.dSdp = X[:,D[i]:D[i+1]].reshape(Dx*2+Dp)
        i+= 1
        do.dSds = X[:,D[i]:D[i+1]].reshape(Dx*4); i+= 1

        do.dMdxx = X[:,D[i]:D[i+1]].reshape(Dx*3); i+= 1
        do.dMdxu = X[:,D[i]:D[i+1]].reshape(Dx*2+Du); i+= 1
        do.dMduu = X[:,D[i]:D[i+1]].reshape(Dx+Du*2); i+= 1
        if self.num_param > 0: 
            do.dMdxp = X[:,D[i]:D[i+1]].reshape(Dx*2+Dp); i+= 1
            do.dMdup = X[:,D[i]:D[i+1]].reshape(Dx+Du+Dp); i+= 1
            do.dMdpp = X[:,D[i]:D[i+1]].reshape(Dx+Dp*2); i+= 1
        
        return x, S, do

    def _ode_moment_match (self, x, S, u, grad=False):
        input_mean, _ = self.get_input_mean_and_cov(x, S, u)

        dM, do = self.f( *input_mean, grad=True )
        dS     = self.x_covar + np.matmul(do.dMdx, S) + np.matmul(S, do.dMdx.T)
        if not grad:
            return dM, dS, do

        I = np.eye(self.num_states)[:,None,:,None] \
            * np.eye(self.num_states)[None,:,None,:]
        do.dSdx  = np.einsum("kj,ijn->kin", S, do.dMdxx)
        do.dSdx += np.transpose(do.dSdx, [1,0,2])
        do.dSdu  = np.einsum("kj,ijn->kin", S, do.dMdxu)
        do.dSdu += np.transpose(do.dSdu, [1,0,2])
        do.dSds  = np.einsum('ij,jkmn->ikmn',do.dMdx,I)
        do.dSds += np.transpose(do.dSds, [1,0,3,2])
        if self.num_param > 0:
            do.dSdp  = np.einsum("kj,ijn->kin", S, do.dMdxp)
            do.dSdp += np.transpose(do.dSdp, [1,0,2])

        return dM, dS, do

    def _predict_x_dist (self, xk, Sk, u, cross_cov=False, grad=False, T=(0,1)):
        if cross_cov:
            raise NotImplementedError(
                'Cross covariance not implemented for continuous time models')
        ode = lambda t, x: self._x_dist_ode(t,x,u,grad=grad)
        X   = self._ode_vector_merge(xk, Sk, grad=grad)
        #T   = self._get_time_steps()
        #t_p = np.array(T)
        #n_tp = len(t_p)-1
        #print(T)
        #T   = (T[0], T[-1])
        #print(T)
        X   = solve_ivp(ode, T, X)
        #print(X)
        X = X['y'][:,-1]
        #print(X)
        M, S, do = self._ode_vector_unmerge(X, grad=grad)
        M, S, do = self.control_and_parameter_uncertainty(M, S, do, grad=grad, T=T)
        
        #M = np.zeros((len(t_p)-1,) + xk.shape)
        #S = np.zeros((len(t_p)-1,) + Sk.shape)
        #if grad:
        #    do = LatentStateDerivatives(self, num_test_points=n_tp)
        #for i in range(n_tp):
        #    M[i,:], S[i,:,:], dok = self._ode_vector_unmerge(X[:,i+1], grad=grad)
        #    M[i,:], S[i,:,:], dok = self.control_and_parameter_uncertainty(M[i], S[i], dok, grad=grad)
        #    if grad:
        #        do.insert(dok,i)
        return (M, S) if not grad else (M, S, do)

    def control_and_parameter_uncertainty (self, M, S, do, grad=False, T=(0,1)):
        #print(T)
        #print(T[1]-T[0])
        #print(S[1,1])
        S += np.matmul( do.dMdu, np.matmul(self.u_covar, do.dMdu.T) )/(T[1]-T[0])
        if self.num_param > 0:
            S += np.matmul( do.dMdp, np.matmul(self.p_covar, do.dMdp.T) )/(T[1]-T[0])
        #print(S[1,1])
        
        if grad:
            # Matrix multiplication
            def mul (s1, s2, i, j=None):
                reverse = len(s2) > len(s1)
                if reverse:
                    s1, s2 = s2, s1
                a1 = getattr(do, s1)
                a2 = getattr(do, s2)
                S  = getattr(self, '%s_covar'%s2[-1])
                o  = 'ab' if s1[-1] == s2[-1] else 'ba'
                j  = i if j is None else j
                if reverse:
                    return np.einsum('c,cb,%s->a'%o, a2[i], S, a1[j])
                return np.einsum('%s,bc,c->a'%o, a1[i], S, a2[j])

            def mul2 (s1, s2, i, j):
                return mul(s1,s2,i,j) + mul(s2,s1,i,j)
            
            for i in range(self.num_states):
                # Input uncertainty
                do.dSdx[i,i] += 2*(mul('dMdxu','dMdu',i) + mul('dMdxp','dMdp',i))
                do.dSdu[i,i] += 2*(mul('dMduu','dMdu',i) + mul('dMdup','dMdp',i))
                do.dSdp[i,i] += 2*(mul('dMdup','dMdu',i) + mul('dMdpp','dMdp',i))
                for j in range(i+1,self.num_states):
                    # Input uncertainty
                    do.dSdx[i,j] += mul2('dMdxu','dMdu',i,j)
                    do.dSdu[i,j] += mul2('dMduu','dMdu',i,j)
                    do.dSdp[i,j] += mul2('dMdup','dMdu',i,j)
                    # Parameter uncertainty
                    do.dSdx[i,j] += mul2('dMdxp','dMdp',i,j)
                    do.dSdu[i,j] += mul2('dMdup','dMdp',i,j)
                    do.dSdp[i,j] += mul2('dMdpp','dMdp',i,j)
                    # Symmetry
                    do.dSdx[j,i]  = do.dSdx[i,j]
                    do.dSdu[j,i]  = do.dSdu[i,j]
                    do.dSdp[j,i]  = do.dSdp[i,j]
        return M, S, do

        
    def _x_dist_ode (self, t, X, U, grad=False):
        x, S, do = self._ode_vector_unmerge(X, grad=grad)

        # Input
        if not (isinstance(U, np.ndarray) or callable(U)):
            raise ValueError('Control input not array or callable')
        u = U if isinstance(U, np.ndarray) else U(t)

        x, S, domm = self._ode_moment_match(x, S, u, grad=grad)

        # Matrix multiplication
        def mul (s1, s2):
            a1, a2 = getattr(domm,s1), getattr(do,s2)
            t1 = {'M':'a'}.get(s1[1],'ab')
            t2 = {'x':'i', 's':'ij'}[s1[3]]
            t3 = {'x':'c', 'u':'c', 'p':'c'}.get(s2[3],'cd')
            return np.einsum( '%s%s,%s%s->%s%s'%(t1,t2,t2,t3,t1,t3), a1, a2 )
        
        dMdx = mul( 'dMdx', 'dMdx' ) + mul( 'dMds', 'dSdx' )
        dMdu = domm.dMdu + mul( 'dMdx', 'dMdu' ) + mul('dMds','dSdu')
        if self.num_param > 0:
            dMdp = domm.dMdp + mul('dMdx','dMdp') + mul('dMds','dSdp')
        
        if grad:
            dMds = mul( 'dMdx', 'dMds' ) + mul( 'dMds', 'dSds' )
            dSdx = mul( 'dSdx', 'dMdx' ) + mul( 'dSds', 'dSdx' )
            dSds = mul( 'dSdx', 'dMds' ) + mul( 'dSds', 'dSds' )
            dSdu = domm.dSdu + mul( 'dSdx', 'dMdu' ) + mul( 'dSds', 'dSdu' )
            if self.num_param > 0:
                dSdp = domm.dSdp + mul( 'dSdx', 'dMdp' ) + mul( 'dSds', 'dSdp' )

            # Tensor multiplication
            def tensmul (s1, s2):
                a1, a2 = getattr(domm,s1), getattr(do,s2)
                if a1.ndim == 3:
                    return np.einsum( 'aib,ic->abc', a1, a2 )
                return np.einsum( 'ai,ibc->abc', a1, a2 )

            # Tensor multiplication with three terms
            def trimul (s1, s2, s3):
                a1, a2 = getattr(do,s1), getattr(domm,s2)
                a3     = np.eye(a2.shape[2]) if s3 == 'I' else getattr(do,s3)
                return np.einsum( 'ib,aij,jc->abc', a1, a2, a3 )

            # Compute Hessian update
            def hess (a, b):
                dMda, dMdb, dMdab = 'dMd'+a, 'dMd'+b, 'dMd'+a+b
                ddM = trimul(dMda, 'dMdxx', dMdb) + tensmul('dMdx', dMdab)
                if b != 'x':
                    ddM += trimul(dMda, 'dMdx'+b, 'I')
                    if a != 'x':
                        ddM += getattr(domm, dMdab) + tensmul('dMdx'+a, dMdb)
                return ddM

            dMdxx = hess('x','x')
            dMdxu = hess('x','u')
            dMduu = hess('u','u')
            if self.num_param > 0:
                dMdxp = hess('x','p')
                dMdup = hess('u','p')
                dMdpp = hess('p','p')

        do.dMdx = dMdx
        do.dMdu = dMdu
        if self.num_param > 0:
            do.dMdp = dMdp
        if grad:
            do.dMds = dMds
            do.dSdx = dSdx
            do.dSds = dSds
            do.dSdu = dSdu
            if self.num_param > 0:
                do.dSdp = dSdp

            do.dMdxx = dMdxx
            do.dMdxu = dMdxu
            do.dMduu = dMduu
            if self.num_param > 0:
                do.dMdxp = dMdxp
                do.dMdup = dMdup
                do.dMdpp = dMdpp
        
        # Total
        return self._ode_vector_merge(x, S, do, grad=grad)
    
    def _get_time_steps (self, start=0., stop=None):
        if stop is None:
            stop = start + self.step_length
        assert start < stop, 'Start time (%f) >= stop time (%f)'%(start,stop)
        return np.linspace(start, stop, self.time_steps)