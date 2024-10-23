
# Imports
from .common import *

## Grayscales
# RGB to Grayscale average
def rgb_to_grayscale_average(img: ImageData) -> ImageData:
	""" Convert an RGB image to Grayscale using average method\n
	Args:
		img		(ImageData):	RGB image (3D array)
	Returns:
		(ImageData): Grayscale image (2D array)
	"""
	return ImageData(np.mean(img.data, axis=0), "Grayscale (average)")

# RGB to Grayscale (norm 709)
def rgb_to_grayscale_norm709(img: ImageData) -> ImageData:
	""" Convert an RGB image to Grayscale using norm 709 method\n
	Args:
		img		(ImageData):	RGB image (3D array)
	Returns:
		(ImageData): Grayscale image (2D array)
	"""
	return ImageData(np.sum(img * NORM709, axis=0), "Grayscale (norm 709)")

# RGB to Grayscale (norm 601)
def rgb_to_grayscale_norm601(img: ImageData) -> ImageData:
	""" Convert an RGB image to Grayscale using norm 601 method\n
	Args:
		img	(ImageData):	RGB image (3D array)
	Returns:
		(ImageData): Grayscale image (2D array)
	"""
	return ImageData(np.sum(img * NORM601, axis=0), "Grayscale (norm 601)")

