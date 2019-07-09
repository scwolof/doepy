# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:05:38 2019

@author: edusc_000
"""


import numpy as np 
from doepy.optimize import slsqp
from doepy.optimize import LogCallback
import matplotlib.pyplot as plt

def SIP (PI, u0, e_g0=0.01, r=4, e_stat=1e-3, e_act=1e-3, debug=True):

    it = 0
    tc_0 = PI.t_const.copy()
    log   = LogCallback()
    # Default option shows the results of each iteration of the optimizer
    # It is possible to hide these results by setting debug=False
    # Default maximum number of iterarions = 30. To change set maxmajiter=50 for example.
    ver_discr = 400
    ptos_ver = np.linspace(PI.t0,PI.tf,ver_discr, True)
    out = None
    while 1:
        PI.update_t(tc_0)
        PI.update_z_bounds(e_g0)
        it += 1
        out = slsqp(PI, u0, log_callback=log, debug=debug, ftol=1e-5)
        U_opt = out['u']
        u0 = U_opt
        print( out['message'] )
        print( 'Iteration: ' + str(it))
        PI.update_t(ptos_ver)
        PI.update_z_bounds(-e_g0)

        f, ci = PI(U_opt, grad=False)
        di = ci.reshape(ver_discr,2*PI.num_meas)
        ind = np.unravel_index(np.argmin(di, axis=None), di.shape)
        minc = di[ind]
        Z,S = PI.simulate(U_opt)
        z = Z[:,0,1]
        s = 1.64*np.sqrt( S[:,0,1,1] )
        plt.plot(PI.t_all,z,'b')
        plt.plot(PI.t_all,0.5*np.ones(len(PI.t_all)),'r')
        plt.fill_between(PI.t_all,z-s,z+s,color='b',alpha=0.2)
        plt.show()
        print('Max constraint:' + str(minc))
        if minc < 0:
            ti_min = PI.t_const[ind[0]]
            print('Violation found.')
            print('Max violation at:'+str(ti_min))
            tc_0 = np.unique(np.concatenate((tc_0,ti_min),axis=None))
            tc_0.sort()
            print(tc_0)
        else:
            print('No violations found.')
            if e_g0 > e_act:
                print('Update of restriction.')
                e_g0 = e_g0/r
                print('New restriction:' + str(e_g0))
            else:
                print('Feasible optimal solution found.')
                break
    return out
