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

from .derivatives import Derivatives
from .gp_derivatives import d_pred_d_x, d2_m_d_x2
from .taylor_moment_match import taylor_moment_match

def gp_taylor_moment_match (gps, m, s, grad=False):
	E, D  = len(gps), len(m)
	M, S  = np.zeros(E), np.zeros((E,E))
	
	for e, gp in enumerate(gps):
		tmp    = gp.predict_noiseless(m[None,:])
		M[e]   = tmp[0][0,0]
		S[e,e] = tmp[1][0,0]

	dMdm, dSdm = d_pred_d_x(gps, m, diag=True)
	ddM        = None if not grad else d2_m_d_x2(gps, m)
	_S, V, do  = taylor_moment_match(s, dMdm, ddM, grad)

	S += _S
	if not grad:
		M, S, V, None

	do.dSdx += dSdm
	return M, S, V, do