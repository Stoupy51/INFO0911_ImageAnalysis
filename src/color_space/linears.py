
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
	y: np.ndarray = np.sum(image * RGB_TO_YIQ_MATRIX[0].reshape(3, 1, 1), axis=0)
	i: np.ndarray = np.sum(image * RGB_TO_YIQ_MATRIX[1].reshape(3, 1, 1), axis=0)
	q: np.ndarray = np.sum(image * RGB_TO_YIQ_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((y, i, q), axis=0)

# YIQ to RGB
def yiq_to_rgb(image: np.ndarray) -> np.ndarray:
	""" Convert an YIQ image to RGB color space\n
	Args:
		image	(np.ndarray):	YIQ image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	"""
	r: np.ndarray = np.sum(image * YIQ_TO_RGB_MATRIX[0].reshape(3, 1, 1), axis=0)
	g: np.ndarray = np.sum(image * YIQ_TO_RGB_MATRIX[1].reshape(3, 1, 1), axis=0)
	b: np.ndarray = np.sum(image * YIQ_TO_RGB_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((r, g, b), axis=0)


# RGB to YUV
def rgb_to_yuv(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to YUV color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): YUV image (3D array)
	"""
	y: np.ndarray = np.sum(image * RGB_TO_YUV_MATRIX[0].reshape(3, 1, 1), axis=0)
	u: np.ndarray = np.sum(image * RGB_TO_YUV_MATRIX[1].reshape(3, 1, 1), axis=0)
	v: np.ndarray = np.sum(image * RGB_TO_YUV_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((y, u, v), axis=0)

# YUV to RGB
def yuv_to_rgb(image: np.ndarray) -> np.ndarray:
	""" Convert an YUV image to RGB color space\n
	Args:
		image	(np.ndarray):	YUV image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	"""
	r: np.ndarray = np.sum(image * YUV_TO_RGB_MATRIX[0].reshape(3, 1, 1), axis=0)
	g: np.ndarray = np.sum(image * YUV_TO_RGB_MATRIX[1].reshape(3, 1, 1), axis=0)
	b: np.ndarray = np.sum(image * YUV_TO_RGB_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((r, g, b), axis=0)


# RGB to I1I2I3
def rgb_to_i1i2i3(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to I1I2I3 color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): I1I2I3 image (3D array)
	"""
	i1: np.ndarray = np.sum(image * RGB_TO_I1I2I3_MATRIX[0].reshape(3, 1, 1), axis=0)
	i2: np.ndarray = np.sum(image * RGB_TO_I1I2I3_MATRIX[1].reshape(3, 1, 1), axis=0)
	i3: np.ndarray = np.sum(image * RGB_TO_I1I2I3_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((i1, i2, i3), axis=0)

# I1I2I3 to RGB
def i1i2i3_to_rgb(image: np.ndarray) -> np.ndarray:
	""" Convert an I1I2I3 image to RGB color space\n
	Args:
		image	(np.ndarray):	I1I2I3 image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	"""
	r: np.ndarray = np.sum(image * I1I2I3_TO_RGB_MATRIX[0].reshape(3, 1, 1), axis=0)
	g: np.ndarray = np.sum(image * I1I2I3_TO_RGB_MATRIX[1].reshape(3, 1, 1), axis=0)
	b: np.ndarray = np.sum(image * I1I2I3_TO_RGB_MATRIX[2].reshape(3, 1, 1), axis=0)
	return np.stack((r, g, b), axis=0)


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

