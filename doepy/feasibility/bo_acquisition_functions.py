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
from scipy.stats import multivariate_normal as mvn

def _EI (m, s2):
    s = np.sqrt(s2)
    return s / np.sqrt(2*np.pi) * mvn.pdf(m / s)

def _dEI (m, s2, dm, ds2):
    c2 = m / s2 
    c1 = (1 + c2 * m) / (2. * s2)
    E  = _EI(m, s2)
    return ( c1[:,None]*ds2 - c2[:,None]*dm ) * E[:,None]

def expected_improvement (x, model, grad=False):
    if x.ndim == 1:
        x = x[None,:]
    m,s2 = model.predict_noiseless( x )
    m,s2 = m[:,0], s2[:,0]
    if not grad:
        return _EI(m, s2)
    dm,ds2 = model.predictive_gradients( x )
    dm,ds2 = dm.reshape((len(m),-1)), ds2.reshape((len(m),-1))
    return _EI(m, s2), _dEI(m, s2, dm, ds2)