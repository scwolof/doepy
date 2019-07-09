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

from .utils import assert_is_instance, assert_is_shape

class Derivatives:
    def __init__ (self, num_out, num_inputs=None, num_param=None, num_states=None, 
        num_test_points=None):

        self.has_inputs = num_inputs is not None and num_inputs > 0
        self.has_param  = num_param is not None and num_param > 0
        self.has_states = num_states is not None and num_states > 0

        # Predictive mean
        self.M = ( num_out, )
        if num_test_points is not None:
            self.M = ( num_test_points, ) + self.M
        
        # Predictive covariance
        self.S = self.M + self.M

        # Predictive input-output covariance
        if self.has_states:
            self.V = ( num_states, ) + self.M

        # Control input
        if self.has_inputs:
            self.u  = ( num_inputs, )
            self.uu = ( num_inputs, num_inputs )

        # Model parameter
        if self.has_param:
            self.p  = ( num_param, )
            self.pp = ( num_param, num_param )
            if self.has_inputs:
                self.up = ( num_inputs, num_param )

        # Latent state mean and covariance
        if self.has_states:
            self.x  = ( num_states, ) 
            self.xx = ( num_states, num_states )
            self.s  = ( num_states, num_states )
            if self.has_inputs:
                self.xu = ( num_states, num_inputs )
            if self.has_param:
                self.xp = ( num_states, num_param )

    def insert (self, do, n):
        D = self.__dict__
        for d in D:
            if d in do.__dict__:
                t    = getattr(self,d)
                if not isinstance(t, np.ndarray):
                    continue
                t[n] = getattr(do,d)

    def _assert_has_derivative (self, x):
        if 'u' in x:
            self._assert_has_given_derivative(self.has_inputs, 'input')
        if 'p' in x:
            self._assert_has_given_derivative(self.has_param, 'model parameter')
        if 'x' in x:
            self._assert_has_given_derivative(self.has_states, 'latent state')

    def _assert_has_given_derivative (self, attr, name):
        if not attr:
            raise AssertionError('Object does not have %s derivatives'%name)

    def _get_shape (self, X, x):
        return getattr(self, X) + getattr(self, x)

    def _get_derivative (self, X, x):
        self._assert_has_derivative(x)

        name = '_d%sd%s'%(X,x)
        if hasattr(self, name):
            return getattr(self, name)
        return np.zeros( self._get_shape(X, x) )

    def _set_derivative (self, X, x, value):
        name = 'd%sd%s'%(X,x)
        assert_is_instance(value, np.ndarray, name)
        assert_is_shape(value, self._get_shape(X, x))
        setattr(self, '_%s'%name, value)

    """
    Control input derivatives
    """
    @property
    def dMdu (self):
        return self._get_derivative('M', 'u')
    @dMdu.setter
    def dMdu (self, value):
        self._set_derivative('M', 'u', value)

    @property
    def dSdu (self):
        return self._get_derivative('S', 'u')
    @dSdu.setter
    def dSdu (self, value):
        self._set_derivative('S', 'u', value)

    @property
    def dVdu (self):
        return self._get_derivative('V', 'u')
    @dVdu.setter
    def dVdu (self, value):
        self._set_derivative('V', 'u', value)

    """
    Model parameter derivatives
    """
    @property
    def dMdp (self):
        return self._get_derivative('M', 'p')
    @dMdp.setter
    def dMdp (self, value):
        self._set_derivative('M', 'p', value)

    @property
    def dSdp (self):
        return self._get_derivative('S', 'p')
    @dSdp.setter
    def dSdp (self, value):
        self._set_derivative('S', 'p', value)

    @property
    def dVdp (self):
        return self._get_derivative('V', 'p')
    @dVdp.setter
    def dVdp (self, value):
        self._set_derivative('V', 'p', value)

    """
    Latent state mean derivatives
    """
    @property
    def dMdx (self):
        return self._get_derivative('M', 'x')
    @dMdx.setter
    def dMdx (self, value):
        self._set_derivative('M', 'x', value)

    @property
    def dSdx (self):
        return self._get_derivative('S', 'x')
    @dSdx.setter
    def dSdx (self, value):
        self._set_derivative('S', 'x', value)

    @property
    def dVdx (self):
        return self._get_derivative('V', 'x')
    @dVdx.setter
    def dVdx (self, value):
        self._set_derivative('V', 'x', value)

    """
    Latent state covariance derivatives
    """
    @property
    def dMds (self):
        return self._get_derivative('M', 's')
    @dMds.setter
    def dMds (self, value):
        self._set_derivative('M', 's', value)

    @property
    def dSds (self):
        return self._get_derivative('S', 's')
    @dSds.setter
    def dSds (self, value):
        self._set_derivative('S', 's', value)

    @property
    def dVds (self):
        return self._get_derivative('V', 's')
    @dVds.setter
    def dVds (self, value):
        self._set_derivative('V', 's', value)

    """
    Hessians
    """
    @property
    def dMduu (self):
        return self._get_derivative('M', 'uu')
    @dMduu.setter
    def dMduu (self, value):
        self._set_derivative('M', 'uu', value)

    @property
    def dMdpp (self):
        return self._get_derivative('M', 'pp')
    @dMdpp.setter
    def dMdpp (self, value):
        self._set_derivative('M', 'pp', value)

    @property
    def dMdxx (self):
        return self._get_derivative('M', 'xx')
    @dMdxx.setter
    def dMdxx (self, value):
        self._set_derivative('M', 'xx', value)

    @property
    def dMdup (self):
        return self._get_derivative('M', 'up')
    @dMdup.setter
    def dMdup (self, value):
        self._set_derivative('M', 'up', value)

    @property
    def dMdxp (self):
        return self._get_derivative('M', 'xp')
    @dMdxp.setter
    def dMdxp (self, value):
        self._set_derivative('M', 'xp', value)

    @property
    def dMdxu (self):
        return self._get_derivative('M', 'xu')
    @dMdxu.setter
    def dMdxu (self, value):
        self._set_derivative('M', 'xu', value)