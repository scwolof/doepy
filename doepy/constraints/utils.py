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

from ..utils import assert_is_instance, assert_is_shape


def bounds_to_linear_constraints (bounds):
    """
    Convert (min, max) bounds to linear constraints system
        A * x - b >= 0

    Inputs:
        bounds      [d, 2] numpy.ndarray, list or tuple.
                    Each entry bounds[i] gives has (min, max) for dim. i

    Returns:
        A       [2*d, d] numpy.ndarray
        b       [2*d,  ]  numpy.ndarray
    """
    assert_is_instance(bounds, (list, tuple, np.ndarray), 'bounds')

    bounds = np.asarray(bounds)
    d = bounds.shape[0]
    assert d > 0, 'Length of bounds must be > 0, got %d'%d
    assert_is_shape(bounds, (d, 2), 'bounds')

    """
    A
    matrix of the form
        |  1  0  ... |
        | -1  0  ... |
        |  0  1  ... |
        |  0 -1  ... |
        |  ...   ... |
    """
    A = np.zeros((2*d, d))
    for i in range(d):
        A[2*i,   i] =  1
        A[2*i+1, i] = -1

    """
    b
    vector of the form
        |  dim_0_lower |
        |  dim_0_upper |
        |  dim_1_lower |
        |     ...      |
    """
    b = np.zeros(2*d)
    for i in range(d):
        b[2*i]   =  bounds[i, 0]
        b[2*i+1] = -bounds[i, 1]

    return A, b