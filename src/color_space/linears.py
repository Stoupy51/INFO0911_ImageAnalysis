
# Imports
from src.color_space.grayscale import NORM601
import numpy as np

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

