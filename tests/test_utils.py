
import numpy as np
import pytest

from doepy import utils

# Positive def symmetric matrix
A = np.array([[2.5, 1], [1, 1.7]])
# Symmetric, not positive definite
B = np.array([[-1, 1], [1, 1.7]])
# Not symmetric
C = np.array([[2.5, 1.1], [1, 1.7]])
# Not square matrix
D = np.array([[2.5, 1, 3], [1, 1.7, 2]])
# Not matrix
E = [[3, 2], [2, 1.]]
# Tensor
F = np.array([[[3,2,1],[2,1,0]], [[0,1,2],[1,2,3]]])
# List
G = ((1,2,3),(4,5,6))
# Not matrix, list or tuple
H = np.random


class TestUtils:

	def test_assert_square_matrix (self):
		# A: square
		utils.assert_square_matrix(A)

		# B: square
		utils.assert_square_matrix(B)

		# C: square
		utils.assert_square_matrix(C)

		# D: not square
		with pytest.raises(AssertionError):
			utils.assert_square_matrix(D)
		
		# None: not matrix
		with pytest.raises(ValueError):
			utils.assert_square_matrix(None)
		
		# E: not matrix
		with pytest.raises(ValueError):
			utils.assert_square_matrix(E)
		
		# F: tensor
		with pytest.raises(AssertionError):
			utils.assert_square_matrix(F)

	def test_assert_symmetric_matrix (self):
		# A: symmetric
		utils.assert_symmetric_matrix(A)

		# B: symmetric
		utils.assert_symmetric_matrix(B)

		# C: not symmetric
		with pytest.raises(AssertionError):
			utils.assert_symmetric_matrix(C)

	def test_assert_pos_def (self):
		# A: pos def
		utils.assert_pos_def(A)

		# B: not pos def
		with pytest.raises(AssertionError):
			utils.assert_pos_def(B)

	def test_assert_is_shape (self):
		# A: (2,2) matrix
		utils.assert_is_shape(A, (2,2))
		utils.assert_is_shape(A, [2,2])
		with pytest.raises(AssertionError):
			utils.assert_is_shape(A, (3,2))
		with pytest.raises(AssertionError):
			utils.assert_is_shape(A, (3,2,1))
		with pytest.raises(ValueError):
			utils.assert_is_shape(A, H)

		# E: (2,2) tuple
		with pytest.raises(ValueError):
			utils.assert_is_shape(E, (2,2))

		# None inputs
		with pytest.raises(ValueError):
			utils.assert_is_shape(E, None)
		with pytest.raises(ValueError):
			utils.assert_is_shape(None, (2,2))

		# F: (2,2,3) tensor
		utils.assert_is_shape(F, (2,2,3))
		with pytest.raises(AssertionError):
			utils.assert_is_shape(F, (2,2))
		with pytest.raises(AssertionError):
			utils.assert_is_shape(F, (2,3,2))
		with pytest.raises(AssertionError):
			utils.assert_is_shape(F, (2,2,3,1))


	def test_assert_equal_shape (self):
		# A, B: equal shapes
		utils.assert_equal_shape(A, B)
		# A, D: not equal shapes
		with pytest.raises(AssertionError):
			utils.assert_equal_shape(A, D)
		# A, F: not equal shapes
		with pytest.raises(AssertionError):
			utils.assert_equal_shape(A, F)

		# Non-numpy.ndarray inputs
		with pytest.raises(ValueError):
			utils.assert_equal_shape(E, A)
		with pytest.raises(ValueError):
			utils.assert_equal_shape(A, E)
		with pytest.raises(ValueError):
			utils.assert_equal_shape(A, G)
		with pytest.raises(ValueError):
			utils.assert_equal_shape(G, A)
		with pytest.raises(ValueError):
			utils.assert_equal_shape(A, H)
		with pytest.raises(ValueError):
			utils.assert_equal_shape(H, A)


	def test_assert_not_none (self):
		# A: not None
		utils.assert_not_none(A)
		# None
		with pytest.raises(AssertionError):
			utils.assert_not_none(None)


	def test_assert_is_instance (self):
		# A: numpy.ndarray
		utils.assert_is_instance(A, np.ndarray)
		utils.assert_is_instance(A, (np.ndarray,))
		utils.assert_is_instance(A, (np.ndarray,list))
		with pytest.raises(AssertionError):
			utils.assert_is_instance(A, list)

		# None
		with pytest.raises(AssertionError):
			utils.assert_is_instance(None, np.ndarray)
		with pytest.raises(ValueError):
			utils.assert_is_instance(A, None)