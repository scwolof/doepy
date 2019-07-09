
import numpy as np
from scipy.interpolate import interp1d

class dop:
    def __init__ (self, models, num_steps, div_criterion, u_bounds,
                  u_constraints=[], z_constraints=[], t_const=[], t0=0, tf=None):
        self.models      = models
        self.num_models  = len( models )
        self.num_meas    = models[0].num_meas
        self.num_steps   = num_steps
        self.divergence  = div_criterion
        self.u_bounds    = u_bounds.copy()
        self.num_control = u_bounds.shape[0]
        self.t0 = t0
        self.t_inputs    = np.linspace(t0,tf,num_steps, False).transpose()
        
        # definition of final time
        if tf is None:
            self.tf = num_steps
        else:
            self.tf = tf
        
        # constraint points
        if len(t_const) == 0:
            self.t_const     = np.linspace(t0,tf,num_steps)
        else:
            self.t_const     = t_const
        
        # number of constraint points
        self.n_tconst    = len(self.t_const)
        
        # integration points
        self.t_all = np.unique(np.concatenate((self.t_const,self.t_inputs)))
        self.t_all.sort()
        
        # total points
        self.n_tall = len(self.t_all)

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
            self.num_constraints += const.sum_num_constraints(self.n_tconst,self.num_models)
                                    
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


    def __call__ (self, u_flat, grad=True):
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
            c, dc = const( U, grad=grad )
            L     = const.num_constraints(self.num_steps)
            C[ i_c: i_c+L ]    = c
            dCdU[ i_c: i_c+L ] = dc
            i_c += L

        # Initial states
        x, s, dxdU, dsdU = [], [], [], []
        for model in self.models:
            x.append( model.x0 )
            s.append( model.x0_covar )
            if grad:
                dxdU.append( np.zeros((N, model.num_states, D)))
                dsdU.append( np.zeros((N, model.num_states, model.num_states, D)))
            else:
                dxdU.append( [] )
                dsdU.append( [] )
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
                   
        
        # Controls at time ti
        ui = interp1d(self.t_inputs, U, kind='previous', axis=0, fill_value='extrapolate')
        # Constraint points count
        n_ck = 0
        nu = 0
        u0 = U[0]
        
        # Iterate over control sequence
        for n, ti in enumerate( self.t_all ):
            #print(ti)
            dZdU.fill(0.)
            dSdU.fill(0.)
            u = ui(ti)
            if u != u0:
                u0 = u.copy()
                nu += 1
            
            if n == self.n_tall-1:
                break
            else:
                Ti = (ti,self.t_all[n+1])
                    
            # Predictive distributions at time n for model i
            for i, model in enumerate( self.models ):
                if grad:
                    x[i], s[i], dox = model.predict_x_dist(x[i], s[i], u, grad=True, T=Ti)
                    Z[i], S[i], doy = model.predict_z_dist(x[i], s[i], grad=True)
                    for j in range( nu+1 ):
                        dxdU[i][j],dsdU[i][j] = gradchain(dox,dxdU,dsdU,i,j)
                        if j == nu:
                            dxdU[i][j] += dox.dMdu
                            dsdU[i][j] += dox.dSdu
                        dZdU[i,:,j],dSdU[i,:,:,j] = gradchain(doy,dxdU,dsdU,i,j)
                else:
                    x[i], s[i] = model.predict_x_dist(x[i], s[i], u, grad=False, T=Ti)
                    Z[i], S[i] = model.predict_z_dist(x[i], s[i])

                # Update latent state constraints
                model.update_x_constraints(x[i], s[i], dxdU[i], dsdU[i])
                
                # State constraint for model i at time n
                if np.in1d(ti, self.t_const):
                    for const in self.z_constraints:
                        L = const.num_constraints()
                        c = const(Z[i], S[i], dZdU[i], dSdU[i], step=n_ck, grad=grad)
                        if grad:
                            c, dcdU    = c
                            dCdU[ i_c: i_c+L ] = dcdU
                        
                        C[ i_c: i_c+L ]    = c
                        i_c += L
                    n_ck += 1
                    
                    
            # Divergence between predictive distributions at time n
            for i, model in enumerate( self.models ):
                # Add measurement noise covariance
                S[i] += model.y_covar
                
                
            
            if grad:
                ftmp, dDdY, dDdS = self.divergence(Z, S, grad=True)
                for j in range( nu+1 ):
                    dfdU[j] = -np.einsum('ij,ijk->k', dDdY, dZdU[:,:,j] ) \
                             + np.einsum('ijk,ijkl->l', dDdS, dSdU[:,:,:,j])
            else:
                ftmp = self.divergence(Z, S, grad=grad)
            f = -ftmp   ## Minimisation -> negative maximisation

        # latent state constraints
        for i, model in enumerate( self.models ):
            res = model.get_x_constraints()
            if not res is None:
                L = res[0].shape[0]
                C[ i_c: i_c+L ]    = res[0]
                dCdU[ i_c: i_c+L ] = res[1]
                i_c += L
        
        if not grad:
            return f, C

        # flatten
        dfdU = dfdU.reshape(u_flat.shape)
        dCdU = dCdU.reshape((-1,) + u_flat.shape)
        return f, C, dfdU, dCdU
    
    def update_t(self, new_t):
        
        # constraint points
        self.t_const     = new_t
        
        # number of constraint points
        self.n_tconst    = len(self.t_const)
        
        # integration points
        self.t_all = np.unique(np.concatenate((self.t_const,self.t_inputs)))
        self.t_all.sort()
        
        # total points
        self.n_tall = len(self.t_all)

        # Compute the number of constraints
        self.num_constraints = 0
        for const in self.u_constraints:
            # Control constraints
            self.num_constraints += const.num_constraints( self.num_steps )

        for const in self.z_constraints:
            # Observed state constraints
            self.num_constraints += const.sum_num_constraints(self.n_tconst,self.num_models)
                                    
        for model in self.models:
            # Latent state constraints - for data-driven surrogate models
            self.num_constraints += self.num_steps * model.num_x_constraints()

    def update_z_bounds(self, e_g0):
        for const in self.z_constraints:
            #const.bounds[:,0] -= e_g0
            const.bounds[:,1] -= e_g0
            
    def simulate(self, u_flat):
        E = self.num_meas
        N = self.num_steps
        M = self.num_models
        D = self.num_control
        U = u_flat.reshape(( N, D ))
        x, s = [], []
        
        for model in self.models:
            x.append( model.x0 )
            s.append( model.x0_covar )
            
        Z = np.zeros(( M, E ))
        S = np.zeros(( M, E, E ))
        
        ui = interp1d(self.t_inputs, U, kind='previous', axis=0, fill_value='extrapolate')
        # Constraint points count
        nu = 0
        u0 = U[0]
        Z_hist = np.zeros((self.n_tall, M, E))
        S_hist = np.zeros((self.n_tall, M, E, E))
        for i, model in enumerate( self.models ):
            Z_hist[0,i,:], S_hist[0,i,:,:] = model.predict_z_dist(x[i], s[i])
            
         # Iterate over control sequence
        for n, ti in enumerate( self.t_all ):
            #print(ti)
            u = ui(ti)
            if u != u0:
                u0 = u.copy()
                nu += 1
            
            if n == self.n_tall-1:
                break
            else:
                Ti = (ti,self.t_all[n+1])
                    
            # Predictive distributions at time n for model i
            for i, model in enumerate( self.models ):
                    x[i], s[i] = model.predict_x_dist(x[i], s[i], u, grad=False, T=Ti)
                    Z[i], S[i] = model.predict_z_dist(x[i], s[i])
                    Z_hist[n+1,i,:] = Z[i]
                    S_hist[n+1,i,:,:] = S[i]
            # Divergence between predictive distributions at time n
            for i, model in enumerate( self.models ):
                # Add measurement noise covariance
                S[i] += model.y_covar
                S_hist[n+1,i] += model.y_covar

        return Z_hist, S_hist
            
