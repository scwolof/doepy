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

from ..derivatives import Derivatives


class LatentStateDerivatives (Derivatives):
	def __init__ (self, model, num_test_points=None):
		d = {'num_out': model.num_states,
		     'num_inputs': model.num_inputs,
		     'num_param': model.num_param,
		     'num_states': model.num_states,
		     'num_test_points':num_test_points}
		super().__init__(**d)


class MeasurementDerivatives (Derivatives):
	def __init__ (self, model, num_test_points=None):
		d = {'num_out': model.num_meas,
		     'num_states': model.num_states,
		     'num_test_points':num_test_points}
		super().__init__(**d)