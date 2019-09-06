
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
        self.param_uncertanty = True

        # Check parametric uncertainty
        if np.count_nonzero(models[0].x0_covar) + np.count_nonzero(models[0].u_covar) + np.count_nonzero(models[0].p_covar) == 0:
            self.param_uncertanty = False
        
        
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

        # Equality Constraints
        eq    = np.zeros( 2*self.num_steps-2 )
        deqdU = np.zeros((2*self.num_steps-2,) + U.shape )        
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
        if not self.param_uncertanty:
            s = None
            dsdU = None

        for model in self.models:
            x.append( model.x0 )
            if self.param_uncertanty:
                s.append( model.x0_covar )
            if grad:
                dxdU.append( np.zeros((N, model.num_states, D)))
                if self.param_uncertanty:
                    dsdU.append( np.zeros((N, model.num_states, model.num_states, D)))
            else:
                dxdU.append( [] )
                if self.param_uncertanty:
                    dsdU.append( [] )
            model.initialise_x_constraints()
       
        Z = np.zeros(( M, E ))
        dZdU = np.zeros(( M, E, N, D))

        if self.param_uncertanty:
            dSdU = np.zeros(( M, E, E, N, D))
            S = np.zeros(( M, E, E ))

        def gradchain (do, dxdU, i, j,dsdU=None):
            if dsdU is not None:
                dMdU = np.matmul( do.dMdx, dxdU[i][j] ) \
                        + np.einsum( 'ijk,jkn->in', do.dMds, dsdU[i][j] )
                dSdU = np.matmul( do.dSdx, dxdU[i][j] ) \
                        + np.einsum( 'imjk,jkn->imn', do.dSds, dsdU[i][j] )
                return dMdU, dSdU
            else:
                dMdU = np.matmul( do.dMdx, dxdU[i][j] ) 
            return dMdU
                   
        
        # Controls at time ti
        ui = interp1d(self.t_inputs, U, kind='previous', axis=0, fill_value='extrapolate')
        # Constraint points count
        n_ck = 0
        nu = 0
        u0 = U[0]
        #eq[0] = u0[0] + u0[1]*(self.t_inputs[1]-self.t_inputs[0]) + u0[2]*((self.t_inputs[1]-self.t_inputs[0])**2)

        # Iterate over control sequence
        for n, ti in enumerate( self.t_all ):


            dZdU.fill(0.)
            if self.param_uncertanty:
                dSdU.fill(0.)
    
            u = ui(ti)
            if nu<self.num_steps-1:
                if ti >= self.t_inputs[nu+1]:
                    eq[2*nu] = u0[0] + u0[1]*(self.t_inputs[nu+1]-self.t_inputs[nu]) + u0[2]*((self.t_inputs[nu+1]-self.t_inputs[nu])**2) + u0[3]*((self.t_inputs[nu+1]-self.t_inputs[nu])**3)-u[0]
                    deqdU[2*nu,nu,:] = [1, (self.t_inputs[nu+1]-self.t_inputs[nu]), ((self.t_inputs[nu+1]-self.t_inputs[nu])**2), ((self.t_inputs[nu+1]-self.t_inputs[nu])**3)] 
                    deqdU[2*nu,nu+1,0] = -1

                    eq[2*nu+1] = u0[1] + 2*u0[2]*(self.t_inputs[nu+1]-self.t_inputs[nu]) + 3*u0[3]*((self.t_inputs[nu+1]-self.t_inputs[nu])**2) -u[1]
                    deqdU[2*nu+1,nu,:] = [0, 1, 2*(self.t_inputs[nu+1]-self.t_inputs[nu]),3*((self.t_inputs[nu+1]-self.t_inputs[nu])**2)]
                    deqdU[2*nu+1,nu+1,1] = -1

                    nu += 1
                    u0 = u.copy()

            
            if n == self.n_tall-1:
                break
            else:
                #Ti = (0,self.t_all[n+1]-ti)
                Ti = (ti-self.t_inputs[nu],self.t_all[n+1]-self.t_inputs[nu])
                    
            # Predictive distributions at time n for model i
            for i, model in enumerate( self.models ):
                if grad:
                    if self.param_uncertanty:
                        x[i], s[i], dox = model.predict_x_dist(x[i], s[i], u, grad=True, T=Ti)
                        Z[i], S[i], doy = model.predict_z_dist(x[i], s[i], grad=True)
                        for j in range( nu+1 ):
                            dxdU[i][j],dsdU[i][j] = gradchain(dox,dxdU,i,j,dsdU=dsdU)
                            if j == nu:
                                dxdU[i][j] += dox.dMdu
                                dsdU[i][j] += dox.dSdu
                            dZdU[i,:,j],dSdU[i,:,:,j] = gradchain(doy,dxdU,i,j,dsdU=dsdU)
                    else:
                        x[i], s, dox = model.predict_x_dist(x[i], None, u, grad=True, T=Ti)
                        Z[i] = x[i]
                        doy = dox
                        for j in range( nu+1 ):
                            dxdU[i][j] = gradchain(dox,dxdU,i,j,dsdU=None)
                            if j == nu:
                                dxdU[i][j] += dox.dMdu
                            dZdU[i,:,j] = gradchain(doy,dxdU,i,j,dsdU=None,)
                else:
                    if self.param_uncertanty:
                        x[i], s[i] = model.predict_x_dist(x[i], s[i], u, grad=False, T=Ti)
                        Z[i], S[i] = model.predict_z_dist(x[i], s[i])
                    else:
                        x[i], s, dox = model.predict_x_dist(x[i], None, u, grad=False, T=Ti)
                        Z[i] = x[i]

                # Update latent state constraints
                #model.update_x_constraints(x[i], s[i], dxdU[i], dsdU[i])
                
                # State constraint for model i at time n
                if np.in1d(ti, self.t_const):
                    for const in self.z_constraints:
                        L = const.num_constraints()
                        if self.param_uncertanty:
                            c = const(Z[i], S[i], dZdU[i], dSdU[i], step=n_ck, grad=grad)
                        else:
                            c = const(Z[i], None, dZdU[i], dSdU=None, step=n_ck, grad=grad)

                        if grad:
                            c, dcdU    = c
                            dCdU[ i_c: i_c+L ] = dcdU
                        
                        C[ i_c: i_c+L ]    = c
                        i_c += L
                    n_ck += 1
                    
                    
            if self.param_uncertanty:
                # Divergence between predictive distributions at time n
                for i, model in enumerate( self.models ):
                    # Add measurement noise covariance
                    S[i] += model.y_covar
                
                
            
            if grad:
                if self.param_uncertanty:
                    ftmp, dDdY, dDdS = self.divergence(Z, S, grad=True)
                    for j in range( nu+1 ):
                        dfdU[j] = -np.einsum('ij,ijk->k', dDdY, dZdU[:,:,j] ) \
                                 + np.einsum('ijk,ijkl->l', dDdS, dSdU[:,:,:,j])
                else:
                    ftmp, dDdY, _ = self.divergence(Z, S=None, grad=True)
                    for j in range( nu+1 ):
                        dfdU[j] = -np.einsum('ij,ijk->k', dDdY, dZdU[:,:,j] ) 
            else:
                if self.param_uncertanty:
                    ftmp = self.divergence(Z, S, grad=grad)
                else:
                    ftmp = self.divergence(Z, S=None, grad=grad)
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
        deqdU = deqdU.reshape((-1,) + u_flat.shape)
        
        return f, eq, C, dfdU, deqdU, dCdU
    
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
            for state in range(self.num_states):
                if const.bounds[state,0] is not None:
                    const.bounds[state,0] += e_g0
                if const.bounds[state,1] is not None:
                    const.bounds[state,1] -= e_g0
            
    def simulate(self, u_flat):
        E = self.num_meas
        N = self.num_steps
        M = self.num_models
        D = self.num_control
        U = u_flat.reshape(( N, D ))
        x, s = [], []
        
        for model in self.models:
            x.append( model.x0 )
            if self.param_uncertanty:
                s.append( model.x0_covar )
            
        Z = np.zeros(( M, E ))
        if self.param_uncertanty:
            S = np.zeros(( M, E, E ))
        
        ui = interp1d(self.t_inputs, U, kind='previous', axis=0, fill_value='extrapolate')
        # Constraint points count
        nu = 0
        u0 = U[0]
        
        Z_hist = np.zeros((self.n_tall, M, E))
        if self.param_uncertanty:
            S_hist = np.zeros((self.n_tall, M, E, E))
        
        for i, model in enumerate( self.models ):
            if self.param_uncertanty:
                Z_hist[0,i,:], S_hist[0,i,:,:] = model.predict_z_dist(x[i], s[i])
            else:
                Z_hist[0,i,:] = x[i]
            
         # Iterate over control sequence
        for n, ti in enumerate( self.t_all ):
            #print(ti)
            u = ui(ti)
            #print(u)
            if nu<self.num_steps-1:
                if ti >= self.t_inputs[nu+1]:
                    #if not (u == u0).all():
                    u0 = u.copy()
                    nu += 1
            
            if n == self.n_tall-1:
                break
            else:
                Ti = (ti-self.t_inputs[nu],self.t_all[n+1]-self.t_inputs[nu])
                    
            # Predictive distributions at time n for model i
            for i, model in enumerate( self.models ):
                if self.param_uncertanty:
                    x[i], s[i] = model.predict_x_dist(x[i], s[i], u, grad=False, T=Ti)
                    Z[i], S[i] = model.predict_z_dist(x[i], s[i])
                    Z_hist[n+1,i,:] = Z[i]
                    S_hist[n+1,i,:,:] = S[i]
                else:
                    x[i], _ = model.predict_x_dist(x[i], None, u, grad=False, T=Ti)
                    Z[i] = x[i]
                    Z_hist[n+1,i,:] = Z[i]

            if self.param_uncertanty:
                # Divergence between predictive distributions at time n
                for i, model in enumerate( self.models ):
                    # Add measurement noise covariance
                    S[i] += model.y_covar
                    S_hist[n+1,i] += model.y_covar

        return (Z_hist, S_hist) if self.param_uncertanty else (Z_hist)

    # Generate a discretized dynamic optimization problem to be solved in GAMS
    #def discretize(self,u_flat, t_disc, method):


            
