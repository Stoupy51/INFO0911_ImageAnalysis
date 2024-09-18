
# Imports
import numpy as np
DIVISION_BY_ZERO: float = 1e-9

# Generate a random vector
def random_vector(size: int, random_distribution: str = "uniform") -> np.ndarray:
	""" Generate a random vector\n
	Args:
		size					(int):	Size of the vector
		random_distribution		(str):	Distribution of the random values
	Returns:
		np.array: Random vector
	Raises:
		ValueError: If random_distribution parameter is invalid
	"""
	if random_distribution == "uniform":
		return np.random.rand(size)
	elif random_distribution == "normal":
		return np.random.randn(size)
	else:
		raise ValueError("Invalid random distribution")

