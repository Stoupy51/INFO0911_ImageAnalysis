
# Imports
from src.color_space.common import *

# RGB to YIQ
def rgb_to_yiq(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to YIQ color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): YIQ image (3D array)
	"""
	y: np.ndarray = np.sum(image * YIQ_MATRIX[0].reshape(3, 1, 1), axis=0)
	i: np.ndarray = np.sum(image * YIQ_MATRIX[1].reshape(3, 1, 1), axis=0)
	q: np.ndarray = np.sum(image * YIQ_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((y, i, q), axis=0)


# RGB to YUV
def rgb_to_yuv(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to YUV color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): YUV image (3D array)
	"""
	y: np.ndarray = np.sum(image * YUV_MATRIX[0].reshape(3, 1, 1), axis=0)
	u: np.ndarray = np.sum(image * YUV_MATRIX[1].reshape(3, 1, 1), axis=0)
	v: np.ndarray = np.sum(image * YUV_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((y, u, v), axis=0)


# RGB to I1I2I3
def rgb_to_i1i2i3(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to I1I2I3 color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): I1I2I3 image (3D array)
	"""
	i1: np.ndarray = np.sum(image * I1I2I3_MATRIX[0].reshape(3, 1, 1), axis=0)
	i2: np.ndarray = np.sum(image * I1I2I3_MATRIX[1].reshape(3, 1, 1), axis=0)
	i3: np.ndarray = np.sum(image * I1I2I3_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((i1, i2, i3), axis=0)

# RGB to RGB Normalized
def rgb_to_rgb_normalized(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to Normalized RGB color space\n
	Equation:
		Normalized RGB = RGB / (R + G + B)
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Normalized RGB image (3D array)
	"""
	return image / np.sum(image, axis=0, keepdims=True)

