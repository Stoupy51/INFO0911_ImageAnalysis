
# Common constants
import numpy as np
from src.image import ImageData
NORM709: np.ndarray = np.array([0.2125, 0.7154, 0.0721]).reshape(3, 1, 1)
NORM601: np.ndarray = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)

RGB_TO_YIQ_MATRIX: np.ndarray = np.array([
	NORM601.flatten(),
	[0.596,-0.274,-0.322],
	[0.211,-0.523, 0.312],
])
YIQ_TO_RGB_MATRIX: np.ndarray = np.linalg.inv(RGB_TO_YIQ_MATRIX)

RGB_TO_YUV_MATRIX: np.ndarray = np.array([
	NORM601.flatten(),
	[-0.147, -0.289, 0.437],
	[0.615, -0.515, -0.100],
])
YUV_TO_RGB_MATRIX: np.ndarray = np.linalg.inv(RGB_TO_YUV_MATRIX)

RGB_TO_I1I2I3_MATRIX: np.ndarray = np.array([
	[ 1/3, 1/3,  1/3],
	[ 1/2, 0,   -1/2],
	[-1/4, 2/4, -1/4],
])
I1I2I3_TO_RGB_MATRIX: np.ndarray = np.linalg.inv(RGB_TO_I1I2I3_MATRIX)

RGB_TO_XYZ_MATRIX: np.ndarray = np.array([
	[0.4124, 0.3576, 0.1805],
	[0.2126, 0.7152, 0.0722],
	[0.0193, 0.1192, 0.9505],
])
XYZ_TO_RGB_MATRIX: np.ndarray = np.array([
	[ 3.2406, -1.5372, -0.4986],
	[-0.9689,  1.8758,  0.0415],
	[ 0.0557, -0.2040,  1.0570],
])

ILLUMINANTS: dict[str, np.ndarray] = {
	'D65': np.array([0.95047, 1.00000, 1.08883]).reshape(3, 1, 1),
	'D50': np.array([0.96422, 1.00000, 0.82521]).reshape(3, 1, 1),
}

