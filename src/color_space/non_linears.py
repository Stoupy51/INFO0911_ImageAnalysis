
# Imports
from src.color_space.common import *

# RGB to HSL, HSV, CMYK, L*a*b, L*u*v		(combinaisons non-linéaires)

# RGB to HSL
def rgb_to_hsl(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to HSL color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): HSL image (3D array)
	Source:
		https://www.rapidtables.com/convert/color/rgb-to-hsl.html
	"""
	# Normalize the image
	normalized: np.ndarray = image / 255

	# Get the maximum and minimum values
	Cmax: np.ndarray = np.max(normalized, axis=0)
	Cmin: np.ndarray = np.min(normalized, axis=0)
	delta: np.ndarray = Cmax - Cmin

	# Lightness calculation
	l: np.ndarray = (Cmax + Cmin) / 2

	# Saturation calculation
	s: np.ndarray = np.where(delta != 0, delta / (1 - np.abs(2 * l - 1)), 0)

	# Hue calculation
	h: np.ndarray = np.zeros_like(delta)
	h = np.where(Cmax == normalized[0], ((normalized[1] - normalized[2]) / delta) % 6, h)	# Red is max
	h = np.where(Cmax == normalized[1], 2 + (normalized[2] - normalized[0]) / delta, h)		# Green is max
	h = np.where(Cmax == normalized[2], 4 + (normalized[0] - normalized[1]) / delta, h)		# Blue is max
	h = np.where(delta == 0, 0, h)	# If delta is 0, then h is 0
	h = (h * 60) % 360	# Convert to degrees

	# Return the HSL image
	return np.stack((h, s, l), axis=0)

# RGB to HSV
def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to HSV color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): HSV image (3D array)
	Source:
		https://www.rapidtables.com/convert/color/rgb-to-hsv.html
	"""
	# Normalize the image
	normalized: np.ndarray = image / 255

	# Get the maximum and minimum values
	Cmax: np.ndarray = np.max(normalized, axis=0)
	Cmin: np.ndarray = np.min(normalized, axis=0)
	delta: np.ndarray = Cmax - Cmin

	# Hue calculation
	h: np.ndarray = np.zeros_like(delta)
	h = np.where(Cmax == normalized[0], ((normalized[1] - normalized[2]) / delta) % 6, h)	# Red is max
	h = np.where(Cmax == normalized[1], 2 + (normalized[2] - normalized[0]) / delta, h)		# Green is max
	h = np.where(Cmax == normalized[2], 4 + (normalized[0] - normalized[1]) / delta, h)		# Blue is max
	h = np.where(delta == 0, 0, h)	# If delta is 0, then h is 0
	h = (h * 60) % 360	# Convert to degrees

	# Saturation calculation
	s: np.ndarray = np.zeros_like(delta)
	s[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]

	# Value calculation
	v: np.ndarray = Cmax

	# Return the HSV image
	return np.stack((h, s, v), axis=0)

# RGB to CMYK
def rgb_to_cmyk(image: np.ndarray) -> np.ndarray:
	""" Convert an RGB image to CMYK color space\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): CMYK image (3D array with 4 channels, shape=(4, height, width))
	Source:
		https://www.rapidtables.com/convert/color/rgb-to-cmyk.html
	"""
	# Normalize the image
	normalized: np.ndarray = image / 255

	# Black key calculation
	k: np.ndarray = 1 - np.max(normalized, axis=0)

	# Cyan, Magenta, Yellow calculation
	c: np.ndarray = (1 - normalized[0] - k) / (1 - k)	# Based on red
	m: np.ndarray = (1 - normalized[1] - k) / (1 - k)	# Based on green
	y: np.ndarray = (1 - normalized[2] - k) / (1 - k)	# Based on blue

	# Return the CMYK image
	return np.stack((c, m, y, k), axis=0)

# RGB to CIELAB
def rgb_to_lab(image: np.ndarray, illuminant: str = 'D65') -> np.ndarray:
	""" Convert an RGB image to CIELAB color space\n
	Args:
		image		(np.ndarray):	RGB image (3D array)
		illuminant	(str):			White point illuminant, either 'D65' or 'D50' (default: 'D65')
	Returns:
		(np.ndarray): CIELAB image (3D array)
	Sources:
		https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ					(RGB to CIEXYZ)
		https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB	(CIEXYZ to CIELAB)
	"""
	# Normalize the image
	normalized: np.ndarray = image / 255

	# RGB to CIEXYZ
	xyz: np.ndarray = np.zeros_like(normalized)
	xyz[0] = normalized[0] * RGB_TO_XYZ_MATRIX[0, 0] + normalized[1] * RGB_TO_XYZ_MATRIX[0, 1] + normalized[2] * RGB_TO_XYZ_MATRIX[0, 2]
	xyz[1] = normalized[0] * RGB_TO_XYZ_MATRIX[1, 0] + normalized[1] * RGB_TO_XYZ_MATRIX[1, 1] + normalized[2] * RGB_TO_XYZ_MATRIX[1, 2]
	xyz[2] = normalized[0] * RGB_TO_XYZ_MATRIX[2, 0] + normalized[1] * RGB_TO_XYZ_MATRIX[2, 1] + normalized[2] * RGB_TO_XYZ_MATRIX[2, 2]

	## CIEXYZ to CIELAB
	# Normalize the CIEXYZ image
	if illuminant not in ILLUMINANTS:
		raise ValueError(f"Unknown illuminant '{illuminant}', must be in {list(ILLUMINANTS.keys())}")
	xyz /= ILLUMINANTS[illuminant]
	x, y, z = xyz

	# Pre-calculations
	DELTA: float = 6 / 29	# Threshold
	DELTA_CUBED: float = DELTA ** 3
	DELTA_MINUS_SQUARED: float = DELTA ** (-2)

	# Luminance calculation
	l: np.ndarray = 116 * np.where(y > DELTA_CUBED, np.cbrt(y), 1/3 * y * DELTA_MINUS_SQUARED + 4/29) - 16

	# A and B calculation
	a: np.ndarray = 500 * (
		np.where(x > DELTA_CUBED, np.cbrt(x), 1/3 * x * DELTA_MINUS_SQUARED + 4/29)
		-
		np.where(y > DELTA_CUBED, np.cbrt(y), 1/3 * y * DELTA_MINUS_SQUARED + 4/29)
	)
	b: np.ndarray = 200 * (
		np.where(y > DELTA_CUBED, np.cbrt(y), 1/3 * y * DELTA_MINUS_SQUARED + 4/29)
		-
		np.where(z > DELTA_CUBED, np.cbrt(z), 1/3 * z * DELTA_MINUS_SQUARED + 4/29)
	)	

	# Return the CIELAB image
	return np.stack((l, a, b), axis=0)

