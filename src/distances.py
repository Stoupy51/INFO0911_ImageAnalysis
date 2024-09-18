
# Imports
import numpy as np
DIVISION_BY_ZERO: float = 1e-9

# Util function to prevent division by zero
def safe_division(x: float|np.ndarray) -> float|np.ndarray:
	""" Prevent division by zero by adding a small value if equal to zero\n
	Args:
		x	(float|np.ndarray):	Value that will divide
	Returns:
		float|np.ndarray: Value divided by x
	"""
	if isinstance(x, np.ndarray):
		return np.array([value if value != 0 else DIVISION_BY_ZERO for value in x])
	else:
		return x if x != 0 else DIVISION_BY_ZERO

# Manhattan (L1)
def distance_manhattan(x: np.ndarray, y: np.ndarray) -> float:
	""" Manhattan distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
	Returns:
		float: Manhattan distance between x and y
	"""
	return np.sum(np.abs(x - y))

# Euclidean (L2)
def distance_euclidean(x: np.ndarray, y: np.ndarray) -> float:
	""" Euclidean distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
	Returns:
		float: Euclidean distance between x and y
	"""
	return np.sqrt(np.sum((x - y) ** 2))

# Tchebyshev (L3 inf)
def distance_tchebyshev(x: np.ndarray, y: np.ndarray) -> float:
	""" Tchebyshev distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
	Returns:
		float: Tchebyshev distance between x and y
	"""
	return np.max(np.abs(x - y))

# Minkowski
def distance_minkowski(x: np.ndarray, y: np.ndarray, p: float = 1.5) -> float:
	""" Minkowski distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
		p	(float):		Order of the Minkowski distance
	Returns:
		float: Minkowski distance between x and y
	"""
	assert p > 0, "p must be greater than 0"
	power: float = 1 / p	# Power for the root
	return np.sum(np.abs(x - y) ** p) ** power

# Histogram Intersection
def distance_histogram_intersection(x: np.ndarray, y: np.ndarray) -> float:
	""" Histogram Intersection distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
	Returns:
		float: Histogram Intersection distance between x and y
	"""
	return np.sum(np.minimum(x, y)) / safe_division(np.sum(y))

# Khi2
def distance_khi2(x: np.ndarray, y: np.ndarray) -> float:
	""" Khi2 distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
	Returns:
		float: Khi2 distance between x and y
	"""
	return np.sum((x - y) ** 2 / safe_division((x + y) ** 2))



