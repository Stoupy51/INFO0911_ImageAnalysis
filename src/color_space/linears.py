
# Imports
from .common import *

# RGB to YIQ
def rgb_to_yiq(img: ImageData) -> ImageData:
	""" Convert an RGB image to YIQ color space\n
	Args:
		img		(ImageData):	RGB image (3D array)
	Returns:
		(ImageData): YIQ image (3D array)
	"""
	y: np.ndarray = np.sum(img.data * RGB_TO_YIQ_MATRIX[0].reshape(3, 1, 1), axis=0)
	i: np.ndarray = np.sum(img.data * RGB_TO_YIQ_MATRIX[1].reshape(3, 1, 1), axis=0)
	q: np.ndarray = np.sum(img.data * RGB_TO_YIQ_MATRIX[2].reshape(3, 1, 1), axis=0)
	return ImageData(np.stack((y, i, q), axis=0), "YIQ")

# YIQ to RGB
def yiq_to_rgb(img: ImageData) -> ImageData:
	""" Convert an YIQ image to RGB color space\n
	Args:
		img		(ImageData):	YIQ image (3D array)
	Returns:
		(ImageData): RGB image (3D array)
	"""
	r: np.ndarray = np.sum(img.data * YIQ_TO_RGB_MATRIX[0].reshape(3, 1, 1), axis=0)
	g: np.ndarray = np.sum(img.data * YIQ_TO_RGB_MATRIX[1].reshape(3, 1, 1), axis=0)
	b: np.ndarray = np.sum(img.data * YIQ_TO_RGB_MATRIX[2].reshape(3, 1, 1), axis=0)
	return ImageData(np.stack((r, g, b), axis=0), "RGB")


# RGB to YUV
def rgb_to_yuv(img: ImageData) -> ImageData:
	""" Convert an RGB image to YUV color space\n
	Args:
		img		(ImageData):	RGB image (3D array)
	Returns:
		(np.ndarray): YUV image (3D array)
	"""
	y: np.ndarray = np.sum(img.data * RGB_TO_YUV_MATRIX[0].reshape(3, 1, 1), axis=0)
	u: np.ndarray = np.sum(img.data * RGB_TO_YUV_MATRIX[1].reshape(3, 1, 1), axis=0)
	v: np.ndarray = np.sum(img.data * RGB_TO_YUV_MATRIX[2].reshape(3, 1, 1), axis=0)
	return ImageData(np.stack((y, u, v), axis=0), "YUV")

# YUV to RGB
def yuv_to_rgb(img: ImageData) -> ImageData:
	""" Convert an YUV image to RGB color space\n
	Args:
		img		(ImageData):	YUV image (3D array)
	Returns:
		(ImageData): RGB image (3D array)
	"""
	r: np.ndarray = np.sum(img.data * YUV_TO_RGB_MATRIX[0].reshape(3, 1, 1), axis=0)
	g: np.ndarray = np.sum(img.data * YUV_TO_RGB_MATRIX[1].reshape(3, 1, 1), axis=0)
	b: np.ndarray = np.sum(img.data * YUV_TO_RGB_MATRIX[2].reshape(3, 1, 1), axis=0)
	return ImageData(np.stack((r, g, b), axis=0), "RGB")


# RGB to I1I2I3
def rgb_to_i1i2i3(img: ImageData) -> ImageData:
	""" Convert an RGB image to I1I2I3 color space\n
	Args:
		img	(ImageData):	RGB image (3D array)
	Returns:
		(ImageData): I1I2I3 image (3D array)
	"""
	i1: np.ndarray = np.sum(img.data * RGB_TO_I1I2I3_MATRIX[0].reshape(3, 1, 1), axis=0)
	i2: np.ndarray = np.sum(img.data * RGB_TO_I1I2I3_MATRIX[1].reshape(3, 1, 1), axis=0)
	i3: np.ndarray = np.sum(img.data * RGB_TO_I1I2I3_MATRIX[2].reshape(3, 1, 1), axis=0)
	return ImageData(np.stack((i1, i2, i3), axis=0), "I1I2I3")

# I1I2I3 to RGB
def i1i2i3_to_rgb(img: ImageData) -> ImageData:
	""" Convert an I1I2I3 image to RGB color space\n
	Args:
		img		(ImageData):	I1I2I3 image (3D array)
	Returns:
		(ImageData): RGB image (3D array)
	"""
	r: np.ndarray = np.sum(img.data * I1I2I3_TO_RGB_MATRIX[0].reshape(3, 1, 1), axis=0)
	g: np.ndarray = np.sum(img.data * I1I2I3_TO_RGB_MATRIX[1].reshape(3, 1, 1), axis=0)
	b: np.ndarray = np.sum(img.data * I1I2I3_TO_RGB_MATRIX[2].reshape(3, 1, 1), axis=0)
	return ImageData(np.stack((r, g, b), axis=0), "RGB")


# RGB to RGB Normalized
def rgb_to_rgb_normalized(img: ImageData) -> ImageData:
	""" Convert an RGB image to Normalized RGB color space\n
	Equation:
		Normalized RGB = RGB / (R + G + B)
	Args:
		img		(ImageData):	RGB image (3D array)
	Returns:
		(ImageData): Normalized RGB image (3D array)
	"""
	return ImageData(img.data / np.sum(img.data, axis=0, keepdims=True), "RGB Normalized")

