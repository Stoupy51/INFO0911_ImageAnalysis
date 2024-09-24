
# Imports
import numpy as np
NORM709: np.ndarray = np.array([0.2125, 0.7154, 0.0721]).reshape(3, 1, 1)
NORM601: np.ndarray = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)

## Grayscales
# RGB to Grayscale average
def rgb_to_grayscale_average(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Grayscale using average method\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Grayscale image (2D array)
	"""
	return np.mean(image, axis=0)

# RGB to Grayscale (norm 709)
def rgb_to_grayscale_norm709(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Grayscale using norm 709 method\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Grayscale image (2D array)
	"""
	return np.sum(image * NORM709, axis=0)

# RGB to Grayscale (norm 601)
def rgb_to_grayscale_norm601(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Grayscale using norm 601 method\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Grayscale image (2D array)
	"""
	return np.sum(image * NORM601, axis=0)

