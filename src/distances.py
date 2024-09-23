
# Imports
import numpy as np
from typing import Callable

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
	assert p != 0, "p must be different than 0"
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
	return np.sum(np.minimum(x, y)) / np.sum(y)

# Khi2
def distance_khi2(x: np.ndarray, y: np.ndarray) -> float:
	""" Khi2 distance between two points\n
	Args:
		x	(np.ndarray):	First vector
		y	(np.ndarray):	Second vector
	Returns:
		float: Khi2 distance between x and y
	"""
	return np.sum((x - y) ** 2 / (x + y) ** 2)


# Constants
DISTANCES_CALLS: dict[str, Callable] = {
	"manhattan":				distance_manhattan,
	"euclidean":				distance_euclidean,
	"tchebyshev":				distance_tchebyshev,
	"minkowski":				distance_minkowski,
	"histogram_intersection":	distance_histogram_intersection
}

