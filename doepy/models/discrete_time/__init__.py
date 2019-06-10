from .model import dtModel
from .linearmodel import dtLinearModel
#from .nonlinearmodel import dtNonLinearModel

try:
	import GPy
except:
	import warnings
	warnings.warn("Could not import GPy - cannot import GP models")
else:
	from .gpmodel import dtGPModel
	from .sparsegpmodel import dtSparseGPModel