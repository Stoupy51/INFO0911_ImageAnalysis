
# Imports
from src.color_space.common import *

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
	if BLAZINGLY_FAST:
		return image[0] * NORM709[0] + image[1] * NORM709[1] + image[2] * NORM709[2]
	else:
		return np.sum(image * NORM709, axis=0)

# RGB to Grayscale (norm 601)
def rgb_to_grayscale_norm601(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Grayscale using norm 601 method\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Grayscale image (2D array)
	"""
	if BLAZINGLY_FAST:
		return image[0] * NORM601[0] + image[1] * NORM601[1] + image[2] * NORM601[2]
	else:
		return np.sum(image * NORM601, axis=0)

