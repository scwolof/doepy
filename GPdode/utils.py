
import numpy as np

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
