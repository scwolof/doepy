
from .core_model import CoreModel
from .linearmodel import LinearModel
from .nonlinearmodel import NonLinearModel
from .candidate_model import CandidateModel

try:
	import GPy
except:
	import warnings
	warnings.warn("Could not import GPy - cannot import GP models")
else:
	from .gpmodel import GPModel
	from .sparsegpmodel import SparseGPModel
