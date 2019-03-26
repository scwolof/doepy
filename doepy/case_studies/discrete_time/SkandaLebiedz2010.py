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
from scipy.integrate import odeint

from ..continuous_time import SkandaLebiedz2010

"""
D. Skanda and D. Lebiedz (2010)
"An optimal experimental design approach to model discrimination
in dynamic biochemical systems"
Bioinformatics 26(7): 939-945

D. Skanda and D. Lebiedz (2013)
"A robust optimization approach to experimental design for model 
discrimination of dynamical systems"
Math Program, Series A 141: 405-433

Comments:
Original case study does not have control inputs.
The experimental design is wrt to measurement time points.
We modify this by introducing controls u1 and u2 that perturb the states.
"""

class Model (SkandaLebiedz2010.Model):
	def __init__ (self, name):
		super().__init__(name)

	def __call__ (self, x, u, p):
		f = lambda x,t: self._ode_func(x,u,p)
		t = np.linspace(0, self.dt, 51)
		X = odeint(f, x, t)
		return X[-1]


class M1 (Model):
	"""
	Allosteric model with positive feedback and linear product sink
	"""
	def __init__ (self):
		super().__init__('M1')
		self._ode_func = SkandaLebiedz2010.M1()
		self.p0        = self._ode_func.p0
		self.p_bounds  = self._ode_func.p_bounds


class M2 (Model):
	"""
	Allosteric model with positive feedback in the absence of cooperativity
	when the product sink is represented by Michaelis-Menten kinetics
	"""
	def __init__ (self):
		super().__init__('M2')
		self._ode_func = SkandaLebiedz2010.M2()
		self.p0        = self._ode_func.p0
		self.p_bounds  = self._ode_func.p_bounds


class DataGen (M1):
	def __init__ (self):
		super().__init__()
		self.true_param = np.array([ 0.889, 2.16, 0.111, 1.76e4 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2()]