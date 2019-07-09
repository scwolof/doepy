# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:14:55 2019

@author: edusc_000
"""

# Standard math package
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify
num_time_steps = 40
from doepy.case_studies.continuous_time import Penicillin as Pen

model = Pen.M1(grad=False)

arg_symbols = symbols(['x:'+str(model.num_states),'u:'+str(model.num_inputs),'p:'+str(model.num_param)])
sym_func = model.s_model(*arg_symbols)
s_all = arg_symbols[0]

t0 = 0
tf = 10
dt = 1

nt = int((tf-t0)/dt)

leq = []
snames = [0 for i in range(model.num_states)]

n_x = 0
states = []
conststates = []
functions = []
scname = []

for k, f in enumerate(sym_func):
    if f == 0:
        name = s_all[k].name +"_0"
        conststates.append(name)
        scname.append((s_all[k].name,name))
    else:
        states.append(s_all[k])
        functions.append(f)
        n_x +=1

snames = [0 for i in range(n_x)]
for name in scname:
    snames.append(name)
varstates = []

for i in range(nt):
    for k,state in enumerate(states):
        varstates.append(state.name + "_" + str(i))
        snames[k] = (state.name, state.name + "_" + str(i))
            
    for j in range(n_x):
        if type(sym_func[j]) == int:
            leq.append(str(states[j])+"_"+str(i+1) +"=="+ str(states[j])+ "_" +str(i) +"+("+ str(sym_func[j]*dt)+")")
        else:
            leq.append(str(states[j])+"_"+str(i+1) +"=="+ str(states[j])+ "_" +str(i) +"+(" + str(sym_func[j].subs(snames)*dt)+")" )