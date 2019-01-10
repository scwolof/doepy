
import numpy as np
import tensorflow as tf

from gpflow import settings


def is_symmetric_matrix (A):
	return isinstance(A, np.ndarray) and A.ndim == 2 and np.array_equal(A, A.T)

def is_pos_def (A):
    if is_symmetric_matrix(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    return False


def block_diagonal(matrices, dtype=settings.float_type):
	r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

	Args:
		matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
			matrices with the same batch dimension).
		dtype: Data type to use. The Tensors in `matrices` must match this dtype.
	Returns:
		A matrix with the input matrices stacked along its main diagonal, having
		shape [..., \sum_i N_i, \sum_i M_i].

	Code by Allen Lavoie (Stack Overflow):
	https://stackoverflow.com/questions/42157781/
	"""
	matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
	blocked_rows = tf.Dimension(0)
	blocked_cols = tf.Dimension(0)
	batch_shape = tf.TensorShape(None)
	for matrix in matrices:
		full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
		batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
		blocked_rows += full_matrix_shape[-2]
		blocked_cols += full_matrix_shape[-1]
	ret_columns_list = []
	for matrix in matrices:
		matrix_shape = tf.shape(matrix)
		ret_columns_list.append(matrix_shape[-1])
	ret_columns = tf.add_n(ret_columns_list)
	row_blocks = []
	current_column = 0
	for matrix in matrices:
		matrix_shape = tf.shape(matrix)
		row_before_length = current_column
		current_column += matrix_shape[-1]
		row_after_length = ret_columns - current_column
		row_blocks.append(tf.pad(
			tensor=matrix,
			paddings=tf.concat(
				[tf.zeros([tf.rank(matrix) - 1, 2], dtype=settings.int_type),
				 [(row_before_length, row_after_length)]],
				axis=0)))
	blocked = tf.concat(row_blocks, -2)
	blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
	return blocked