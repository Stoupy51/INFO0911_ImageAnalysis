
# Imports
import numpy as np

# Util functions to generate a random image
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

def img_to_sliced_rgb(image: np.ndarray) -> np.ndarray:
	""" Slice the image in 3 images (each color)\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Sliced RGB image (3D array)
	"""
	return np.moveaxis(image, -1, 0)

def sliced_rgb_to_img(sliced_image: np.ndarray) -> np.ndarray:
	""" Merge the 3 images in one\n
	Args:
		sliced_image	(np.ndarray):	Sliced RGB image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	"""
	return np.moveaxis(sliced_image, 0, -1)

## TODO:
# 3 RGB vers niveaux de gris
# RGB to YIQ, YUV, I1I2I3, RGB Normalisé	(combinaisons linéaires)
# RGB to HSL, HSV, CMYK, L*a*b, L*u*v		(combinaisons non-linéaires)

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
NORM709: np.ndarray = np.array([0.2125, 0.7154, 0.0721]).reshape(3, 1, 1)
def rgb_to_grayscale_norm709(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Grayscale using norm 709 method\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Grayscale image (2D array)
	"""
	return np.sum(image * NORM709, axis=0)

# RGB to Grayscale (norm 601)
NORM601: np.ndarray = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)
def rgb_to_grayscale_norm601(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Grayscale using norm 601 method\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Grayscale image (2D array)
	"""
	return np.sum(image * NORM601, axis=0)


## Linears
# RGB to YIQ
YIQ_MATRIX: np.ndarray = np.array([
	NORM601.flatten(),
	[0.596,-0.274,-0.322],
	[0.211,-0.523, 0.312],
])
def rgb_to_yiq(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to YIQ color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): YIQ image (3D array)
	"""
	# y: np.ndarray = np.sum(image * YIQ_MATRIX[0].reshape(3, 1, 1), axis=0)
	# i: np.ndarray = np.sum(image * YIQ_MATRIX[1].reshape(3, 1, 1), axis=0)
	# q: np.ndarray = np.sum(image * YIQ_MATRIX[2].reshape(3, 1, 1), axis=0)
	# return np.stack([y, i, q], axis=0)
	return np.stack([np.sum(image * YIQ_MATRIX[i].reshape(3, 1, 1), axis=0) for i in range(3)], axis=0)

# RGB to YUV
YUV_MATRIX: np.ndarray = np.array([
	NORM601.flatten(),
	[-0.147, -0.289, 0.437],
	[0.615, -0.515, -0.100],
]).reshape(3, 3, 1)
def rgb_to_yuv(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to YUV color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): YUV image (3D array)
	"""
	return np.stack([np.sum(image * YUV_MATRIX[i].reshape(3, 1, 1), axis=0) for i in range(3)], axis=0)

# RGB to I1I2I3
I1I2I3_MATRIX: np.ndarray = np.array([
	[ 1/3, 1/3,  1/3],
	[ 1/2, 0,   -1/2],
	[-1/4, 2/4, -1/4]
]).reshape(3, 3, 1)
def rgb_to_i1i2i3(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to I1I2I3 color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): I1I2I3 image (3D array)
	"""
	return np.stack([np.sum(image * I1I2I3_MATRIX[i].reshape(3, 1, 1), axis=0) for i in range(3)], axis=0)

# RGB to RGB Normalized
def rgb_to_rgb_normalized(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Normalized RGB color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Normalized RGB image (3D array)
	"""
	return image / np.sum(image, axis=0, keepdims=True)	# Normalize each pixel

