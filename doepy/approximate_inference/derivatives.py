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

from ..derivatives import Derivatives as DerivativesCore

class Derivatives (DerivativesCore):
	def __init__ (self, num_inputs, num_target):
		super().__init__(num_target, num_states=num_inputs)
		"""
		D - num_input
		E - num_target

		self.dMdx    ( (E, D) )       # output mean by input mean
		self.dMds    ( (E, D, D) )    # output mean by input covariance
		self.dSdx    ( (E, E, D) )    # output covariance by input mean
		self.dSds    ( (E, E, D, D) ) # output covariance by input covar
		self.dVdx    ( (D, E, D) )    # output covariance by input mean
		self.dVds    ( (D, E, D, D) ) # output covariance by input covar
		"""