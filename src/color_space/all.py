
# Imports
import numpy as np

# RGB to Grayscale (average, norm 709, norm 601)
# RGB to YIQ, YUV, I1I2I3, RGB Normalisé	(combinaisons linéaires)
# RGB to HSL, HSV, CMYK, L*a*b, L*u*v		(combinaisons non-linéaires)

# Utility function to generate a random image
def random_image(size: int, seed: int|None = None, maxi: int = 256) -> np.ndarray:
	""" Generate a random image of size `size`x`size`\n
	Args:
		size	(int):	Size of the image
		seed	(int):	Seed for random number generator (default: None)
		maxi	(int):	Maximum value for the randint function
	Returns:
		(np.ndarray): Random image (2D array)
	"""
	if seed is not None:
		np.random.seed(seed)
	return np.random.randint(0, maxi, (3, size, size))

# Convert an image to a sliced RGB image
def img_to_sliced_rgb(image: np.ndarray) -> np.ndarray:
	""" Slice the image in 3 images (each color)\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Sliced RGB image (3D array)
	"""
	return np.moveaxis(image, -1, 0)

# Convert a sliced RGB image to an image
def sliced_rgb_to_img(sliced_image: np.ndarray) -> np.ndarray:
	""" Merge the 3 images in one\n
	Args:
		sliced_image	(np.ndarray):	Sliced RGB image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	"""
	return np.moveaxis(sliced_image, 0, -1)

# Import every function from the other files
from src.color_space.grayscale import *
from src.color_space.linears import *
from src.color_space.non_linears import *

