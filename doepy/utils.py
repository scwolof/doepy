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

def assert_symmetric_matrix (A):
    if not isinstance(A, np.ndarray):
        raise ValueError('Input must be numpy.ndarray, not %s'%type(A))
    if not A.ndim == 2:
        raise AssertionError('Input must be 2D matrix, not %dD'%A.ndim)
    if not np.allclose(A, A.T, rtol=1e-8, atol=1e-10):
        raise AssertionError('Matrix not symmetrical')

def assert_pos_def (A):
    assert_symmetric_matrix(A)
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise AssertionError('Matrix not positive definite')

def assert_is_shape (arr, shape):
    if not isinstance(arr, np.ndarray):
        raise ValueError('Input 1 must be numpy.ndarray, not %s'%type(arr))
    if not isinstance(shape, (list,tuple)):
        raise ValueError('Input 2 must be list or tuple, not %s'%type(shape))
    if arr.shape != tuple(shape):
        raise AssertionError('Array shape mismatch: %s != %s'%(arr.shape, shape))

def assert_equal_shape (array1, array2):
    if not isinstance(array1, np.ndarray):
        raise ValueError('Input must be numpy.ndarray, not %s'%type(array1))
    if not isinstance(array2, np.ndarray):
        raise ValueError('Input must be numpy.ndarray, not %s'%type(array2))
    assert_is_shape(array1, array2.shape)

def assert_not_none (variable, name='<variable_name>'):
    if variable is None:
        raise AssertionError('Variable %s is None'%name)

def assert_is_type (variable, dtype, name='<variable_name>'):
    if not isinstance(variable, dtype):
        raise AssertionError('%s must be of type %s: %s'%(name, dtype, variable))
                    