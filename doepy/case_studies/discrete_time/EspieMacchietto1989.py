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

from ..continuous_time import EspieMacchietto1989

"""
D. Espie and S. Macchietto (1989)
"The optimal design of dynamic experiments"
AIChE J 35(2): 223-229

S. P. Asprey and S. Macchietto (2000)
"Statistical tools for optimal dynamic model building"
Comp Chem Eng 24: 1261-1267

B. H. Chen and S. P. Asprey (2003)
"On the design of optimally informative dynamic experiments for model
discrimination in multiresponse nonlinear situations"
Ind Eng Chem Res 42: 1379-1390
"""

class Model (EspieMacchietto1989.Model):
	def __init__ (self, name):
		super().__init__(name)

	def __call__ (self, x, u, p):
		f = lambda x,t: self._ode_func(x,u,p)
		t = np.linspace(0, self.dt, 26)
		X = odeint(f, x, t)
		return X[-1]
	
	def get_candidate_dict (self):
		d = super().get_candidate_dict()
		d['f'] = self
		d['delta_transition'] = True
		return d


class M1 (Model):
	"""
	Monod kinetics with constant specific death rate
	"""
	def __init__ (self):
		super().__init__('M1')
		self._ode_func = EspieMacchietto1989.M1()
		self.p0        = self._ode_func.p0


class M2 (Model):
	"""
	Contois kinetics with constant specific death rate
	"""
	def __init__ (self):
		Model.__init__(self, 'M2')
		self._ode_func = EspieMacchietto1989.M2()
		self.p0        = self._ode_func.p0


class M3 (Model):
	"""
	Linear specific growth rate
	"""
	def __init__ (self):
		Model.__init__(self, 'M3')
		self._ode_func = EspieMacchietto1989.M3()
		self.p0        = self._ode_func.p0


class M4 (Model):
	"""
	Monod kinetics with constant maintenance energy
	"""
	def __init__ (self):
		Model.__init__(self, 'M4')
		self._ode_func = EspieMacchietto1989.M4()
		self.p0        = self._ode_func.p0


class DataGen (M1):
	def __init__ (self):
		M1.__init__(self)
		self.true_param = np.array([ 0.31, 0.18, 0.55, 0.03 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2(), M3(), M4()]