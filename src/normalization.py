
# Imports
import numpy as np
from typing import Callable

# Normalization functions
def normalize_proba(vector: np.ndarray) -> np.ndarray:
	""" Normalize the input vector to a probability distribution.\n
	Args:
		vector (np.ndarray): The input vector to normalize.
	Returns:
		np.ndarray: The normalized vector.
	"""
	return vector / np.sum(vector)


def normalize_magnitude(vector: np.ndarray) -> np.ndarray:
	""" Normalize the input vector by its magnitude.\n
	Args:
		vector (np.ndarray): The input vector to normalize.
	Returns:
		np.ndarray: The normalized vector.
	"""
	return vector / np.sqrt(np.sum(vector ** 2))


def normalize_min_max(vector: np.ndarray) -> np.ndarray:
	""" Normalize the input vector using min-max normalization.\n
	Args:
		vector (np.ndarray): The input vector to normalize.
	Returns:
		np.ndarray: The normalized vector.
	"""
	min_val: float = np.min(vector)
	max_val: float = np.max(vector)
	return (vector - min_val) / (max_val - min_val)


def normalize_standardization(vector: np.ndarray) -> np.ndarray:
	""" Normalize the input vector using standardization.\n
	Args:
		vector (np.ndarray): The input vector to normalize.
	Returns:
		np.ndarray: The normalized vector.
	"""
	mean: float = np.mean(vector)
	std: float = np.std(vector)
	return (vector - mean) / std


def normalize_rank(vector: np.ndarray) -> np.ndarray:
	""" Normalize the input vector by ranking its elements from 1 to N.\n
	For example, if vector is [10, 5, 8], the ranks will be [3, 1, 2]
	since 10 is the largest (rank 3), 5 is smallest (rank 1), and 8 is in middle (rank 2).\n
	Duplicate values get the same rank. For example [5, 5, 8] becomes [1, 1, 3].\n
	Args:
		vector (np.ndarray): The input vector to normalize.
	Returns:
		np.ndarray: The normalized vector where each element is replaced by its rank.
	"""	
	# Get sorted unique values and their indices
	# unique_vals = [5, 8, 10] (sorted unique values)
	# inverse_indices = [2, 0, 1] (indices to map back to original positions)
	unique_vals, inverse_indices = np.unique(vector, return_inverse=True)
	
	# Create ranks array starting at 1
	# ranks = [1, 2, 3] (ranks for unique values)
	ranks = np.arange(1, len(unique_vals) + 1)
	
	# Map ranks back to original array positions using inverse indices
	# ranks[inverse_indices] = ranks[[2, 0, 1]] = [3, 1, 2]
	# Final result: [3, 1, 2] where 3 is rank of 10, 1 is rank of 5, 2 is rank of 8
	return ranks[inverse_indices]


# Normalization calls
NORMALIZATION_CALLS: dict[str, dict] = {
	"Aucune":			{"function": lambda x: x, "args": {}},
	"Probability":		{"function": normalize_proba, "args": {}},
	"Magnitude":		{"function": normalize_magnitude, "args": {}},
	"Min-Max":			{"function": normalize_min_max, "args": {}},
	"Standardization":	{"function": normalize_standardization, "args": {}},
	"Rank":				{"function": normalize_rank, "args": {}}
}

