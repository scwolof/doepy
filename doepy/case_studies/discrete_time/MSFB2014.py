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

from ..continuous_time import MSFB2014

"""
A. Mesbah, S. Streif, R. Findeisen and R. D. Braatz (2014)
"Active fault diagnosis for nonlinear systems with probabilistic uncertainties"
IFAC Proceedings (2014): 7079-7084
"""

class Model (MSFB2014.Model):
	def __init__ (self, name):
		super().__init__(name)

	def __call__ (self, x, u, p):
		f = lambda x,t: self._ode_func(x,u,p)
		t = np.linspace(0, self.dt, 51)
		X = odeint(f, x, t)
		return X[-1]


class M1 (Model):
	"""
	Nominal scenario (no fault)
	"""
	def __init__ (self):
		super().__init__('M1')
		self._ode_func = MSFB2014.M1()
		self.p0        = self._ode_func.p0


class M2 (Model):
	"""
	Multiplicative actuator fault in inlet pump
	"""
	def __init__ (self):
		super().__init__('M2')
		self._ode_func = MSFB2014.M2()
		self.p0        = self._ode_func.p0


class M3 (Model):
	"""
	Circular leak in tank
	"""
	def __init__ (self):
		super().__init__('M3')
		self._ode_func = MSFB2014.M3()
		self.p0        = self._ode_func.p0


class DataGen (M2):
	def __init__ (self):
		super().__init__()
		self.true_param = np.array([ 0.97, 0.82, 0.96, 0.67 ])

	def __call__ (self, x, u):
		return super().__call__(x, u, self.true_param)


def get ():
	return DataGen(), [M1(), M2(), M3()]