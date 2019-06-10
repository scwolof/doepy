# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:16:37 2019

@author: edusc_000
"""
import math
import scipy.special as spc
import sympy as sp
import numpy as np



def erf(x):
    # save the sign of x
    #sign = 1 if x >= 0 else -1
    
        
    #x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    #y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*sp.exp(-x*x)
        #return sign*y # erf(-x) = -erf(x)
    return y

def erf2(x):
    # save the sign of x
    #sign = 1 if x >= 0 else -1
    
        
    #x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    #y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
        #return sign*y # erf(-x) = -erf(x)
    return y


def cdf_1(x,mean,stdev):
    x_norm = (x-mean)/(stdev**0.5)
    y = erf(x_norm)
    return 0.5*(1+y)
    
def cdf_2(x,mean,stdev):
    x_norm = (x-mean)/(stdev**0.5)
    y = erf2(x_norm)
    return 0.5*(1+y)