
# Imports
from src.color_space.grayscale import NORM601
import numpy as np

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
	s: np.ndarray = np.zeros_like(delta)
	s[delta != 0] = delta / (1 - np.abs(2 * l - 1))

	# Hue calculation
	h: np.ndarray = np.zeros_like(delta)
	h[Cmax == normalized[0]] = (normalized[1] - normalized[2]) / delta[Cmax != 0]		# Red is max
	h[Cmax == normalized[1]] = 2 + (normalized[2] - normalized[0]) / delta[Cmax != 0]	# Green is max
	h[Cmax == normalized[2]] = 4 + (normalized[0] - normalized[1]) / delta[Cmax != 0]	# Blue is max
	h[delta == 0] = 0	# If delta is 0, then h is 0
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
	h[Cmax == normalized[0]] = (normalized[1] - normalized[2]) / delta[Cmax != 0]		# Red is max
	h[Cmax == normalized[1]] = 2 + (normalized[2] - normalized[0]) / delta[Cmax != 0]	# Green is max
	h[Cmax == normalized[2]] = 4 + (normalized[0] - normalized[1]) / delta[Cmax != 0]	# Blue is max
	h[delta == 0] = 0	# If delta is 0, then h is 0
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

