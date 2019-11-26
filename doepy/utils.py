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


def assert_square_matrix (A):
    """
    Check that the input matrix is a square matrix.
    If not, throw error.

    Inputs:
        A   numpy.ndarray
    """
    if not isinstance(A, np.ndarray):
        raise ValueError('Input must be numpy.ndarray, not %s'%type(A))
    if not A.ndim == 2:
        raise AssertionError('Input must be 2D matrix, not %dD'%A.ndim)
    if not A.shape == (A.shape[0], A.shape[0]):
        raise AssertionError('Input not square matrix, shape=(%s,%s)'%A.shape)


def assert_symmetric_matrix (A):
    """
    Check that the input matrix is a symmetric matrix (A = A.T).
    If not, throw error.

    Inputs:
        A   numpy.ndarray
    """
    assert_square_matrix(A)
    if not np.allclose(A, A.T, rtol=1e-8, atol=1e-10):
        raise AssertionError('Matrix not symmetrical')


def assert_pos_def (A):
    """
    Check that the input matrix is positive definite.
    If not, throw error.

    (Heuristic: try to compute Cholesky decomposition)

    Inputs:
        A   [(n1), ..., (nn), d, d] numpy.ndarray
    """
    assert_symmetric_matrix(A)
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise AssertionError('Matrix not positive definite')


def assert_is_shape (A, shape, name='<variable_name>'):
    """
    Check that the input numpy array has given shape.
    If not, throw error.

    Inputs:
        A       numpy.ndarray
        shape   list/tuple of integers. 
                Use -1 for dimensions we do not want to check.
    """
    if not isinstance(A, np.ndarray):
        raise ValueError('Input 1 must be numpy.ndarray, not %s'%type(A))
    if not isinstance(shape, (list,tuple)):
        raise ValueError('Input 2 must be list or tuple, not %s'%type(shape))
    if len(A.shape) != len(shape):
        raise AssertionError("Array '%s' shape mismatch: %s != %s"%(
            name, A.shape, shape))

    # Allow -1 to just copy corresponding dimension in A
    shape_copy = []
    for i,s in enumerate(shape):
        if s == -1:
            shape_copy.append(A.shape[i])
        else:
            shape_copy.append(s)
    shape = tuple(shape_copy)

    if A.shape != shape:
        raise AssertionError("Array '%s' shape mismatch: %s != %s"%(
            name, A.shape, shape))


def assert_equal_shape (array1, array2):
    """
    Check whether two numpy.ndarrays have identical shape.
    If not, throw error.

    Inputs:
        array1   numpy.ndarray
        array2   numpy.ndarray
    """
    if not isinstance(array1, np.ndarray):
        raise ValueError('Input must be numpy.ndarray, not %s'%type(array1))
    if not isinstance(array2, np.ndarray):
        raise ValueError('Input must be numpy.ndarray, not %s'%type(array2))
    assert_is_shape(array1, array2.shape)


def assert_not_none (variable, name='<variable_name>'):
    """
    Check whether the input variable is None.
    If it is, throw error.

    Inputs:
        variable    Variable to check
        name        (str, optional) name of variable used for error message
                    (Default: '<variable_name>')
    """
    if variable is None:
        raise AssertionError('Variable %s is None'%name)


def assert_is_instance (variable, dtype, name='<variable_name>'):
    """
    Check that the input variable is of given type.
    If not, throw error.

    Inputs:
        variable    Variable to check
        dtype       Data type (e.g. int or float) or list of data types
        name        (str, optional) name of variable used for error message
                    (Default: '<variable_name>')
    """
    if dtype is None:
        raise ValueError('dtype must not be None')
    if not isinstance(variable, dtype):
        raise AssertionError('%s must be of type %s: %s'%(name, dtype, variable))