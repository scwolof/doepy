from .model import ctModel

try:
	import GPy
except:
	import warnings
	warnings.warn("Could not import GPy - cannot import GP models")
else:
	from .gpmodel import ctGPModel
	from .sparsegpmodel import ctSparseGPModel