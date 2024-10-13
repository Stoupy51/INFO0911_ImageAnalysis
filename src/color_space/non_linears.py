
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

# HSL to RGB
def hsl_to_rgb(image: np.ndarray) -> np.ndarray:
	""" Convert an HSL image to RGB color space\n
	Args:
		image	(np.ndarray):	HSL image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	Source:
		https://www.rapidtables.com/convert/color/hsl-to-rgb.html
	"""
	# Unpack the image
	h, s, l = image

	# Pre-calculations
	C: np.ndarray = (1 - np.abs(2 * l - 1)) * s
	X: np.ndarray = C * (1 - np.abs((h / 60) % 2 - 1))
	m: np.ndarray = l - C / 2

	# RGB calculation
	r: np.ndarray = np.zeros_like(h)
	r = np.where((0 <= h) & (h < 60), C, r)
	r = np.where((60 <= h) & (h < 120), X, r)
	r = np.where((120 <= h) & (h < 180), 0, r)
	r = np.where((180 <= h) & (h < 240), 0, r)
	r = np.where((240 <= h) & (h < 300), X, r)
	r = np.where((300 <= h) & (h < 360), C, r)
	r += m

	g: np.ndarray = np.zeros_like(h)
	g = np.where((0 <= h) & (h < 60), X, g)
	g = np.where((60 <= h) & (h < 120), C, g)
	g = np.where((120 <= h) & (h < 180), C, g)
	g = np.where((180 <= h) & (h < 240), X, g)
	g = np.where((240 <= h) & (h < 300), 0, g)
	g = np.where((300 <= h) & (h < 360), 0, g)
	g += m

	b: np.ndarray = np.zeros_like(h)
	b = np.where((0 <= h) & (h < 60), 0, b)
	b = np.where((60 <= h) & (h < 120), 0, b)
	b = np.where((120 <= h) & (h < 180), X, b)
	b = np.where((180 <= h) & (h < 240), C, b)
	b = np.where((240 <= h) & (h < 300), C, b)
	b = np.where((300 <= h) & (h < 360), X, b)
	b += m

	# Return the RGB image
	return np.stack((r, g, b), axis=0) * 255


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

# HSV to RGB
def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
	""" Convert an HSV image to RGB color space\n
	Args:
		image	(np.ndarray):	HSV image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	Source:
		https://www.rapidtables.com/convert/color/hsv-to-rgb.html
	"""
	# Unpack the image
	h, s, v = image

	# Pre-calculations
	C: np.ndarray = v * s
	X: np.ndarray = C * (1 - np.abs((h / 60) % 2 - 1))
	m: np.ndarray = v - C

	# RGB calculation
	r: np.ndarray = np.zeros_like(h)
	r = np.where((0 <= h) & (h < 60), C, r)
	r = np.where((60 <= h) & (h < 120), X, r)
	r = np.where((120 <= h) & (h < 180), 0, r)
	r = np.where((180 <= h) & (h < 240), 0, r)
	r = np.where((240 <= h) & (h < 300), X, r)
	r = np.where((300 <= h) & (h < 360), C, r)
	r += m

	g: np.ndarray = np.zeros_like(h)
	g = np.where((0 <= h) & (h < 60), X, g)
	g = np.where((60 <= h) & (h < 120), C, g)
	g = np.where((120 <= h) & (h < 180), C, g)
	g = np.where((180 <= h) & (h < 240), X, g)
	g = np.where((240 <= h) & (h < 300), 0, g)
	g = np.where((300 <= h) & (h < 360), 0, g)
	g += m

	b: np.ndarray = np.zeros_like(h)
	b = np.where((0 <= h) & (h < 60), 0, b)
	b = np.where((60 <= h) & (h < 120), 0, b)
	b = np.where((120 <= h) & (h < 180), X, b)
	b = np.where((180 <= h) & (h < 240), C, b)
	b = np.where((240 <= h) & (h < 300), C, b)
	b = np.where((300 <= h) & (h < 360), X, b)
	b += m

	# Return the RGB image
	return np.stack((r, g, b), axis=0) * 255


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

# CMYK to RGB
def cmyk_to_rgb(image: np.ndarray) -> np.ndarray:
	""" Convert an CMYK image to RGB color space\n
	Args:
		image	(np.ndarray):	CMYK image (3D array with 4 channels, shape=(4, height, width))
	Returns:
		(np.ndarray): RGB image (3D array)
	Source:
		https://www.rapidtables.com/convert/color/cmyk-to-rgb.html
	"""
	# Unpack the image
	c, m, y, k = image

	# RGB calculation
	r: np.ndarray = 255 * (1 - c) * (1 - k)
	g: np.ndarray = 255 * (1 - m) * (1 - k)
	b: np.ndarray = 255 * (1 - y) * (1 - k)

	# Return the RGB image
	return np.stack((r, g, b), axis=0)


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
	xyz: np.ndarray = np.dot(normalized.T, RGB_TO_XYZ_MATRIX.T).T

	## CIEXYZ to CIELAB
	# Normalize the CIEXYZ image
	if illuminant not in ILLUMINANTS:
		raise ValueError(f"Unknown illuminant '{illuminant}', must be in {list(ILLUMINANTS.keys())}")
	xyz /= ILLUMINANTS[illuminant]
	x, y, z = xyz

	# Luminance calculation
	l: np.ndarray = 116 * np.where(y > ((6 / 29) ** 3), np.cbrt(y), 1/3 * y * ((6 / 29) ** (-2)) + 4/29) - 16

	# A and B calculation
	a: np.ndarray = 500 * (
		np.where(x > ((6 / 29) ** 3), np.cbrt(x), 1/3 * x * ((6 / 29) ** (-2)) + 4/29)
		-
		np.where(y > ((6 / 29) ** 3), np.cbrt(y), 1/3 * y * ((6 / 29) ** (-2)) + 4/29)
	)
	b: np.ndarray = 200 * (
		np.where(y > ((6 / 29) ** 3), np.cbrt(y), 1/3 * y * ((6 / 29) ** (-2)) + 4/29)
		-
		np.where(z > ((6 / 29) ** 3), np.cbrt(z), 1/3 * z * ((6 / 29) ** (-2)) + 4/29)
	)	

	# Return the CIELAB image
	return np.stack((l, a, b), axis=0)

# CIELAB to RGB
def lab_to_rgb(image: np.ndarray, illuminant: str = 'D65') -> np.ndarray:
	""" Convert an CIELAB image to RGB color space\n
	Args:
		image		(np.ndarray):	CIELAB image (3D array)
		illuminant	(str):			White point illuminant, either 'D65' or 'D50' (default: 'D65')
	Returns:
		(np.ndarray): RGB image (3D array)
	Sources:
		https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIELAB_to_CIEXYZ	(CIELAB to CIEXYZ)
		https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB					(CIEXYZ to RGB)
	"""
	# Unpack the image
	l, a, b = image

	## CIELAB to CIEXYZ
	# Y calculation
	y: np.ndarray = np.where(l > (3 * ((6 / 29) ** 3)), ((l + 16) / 116) ** 3, l / (29 * ((6 / 29) ** 2)))

	# X calculation
	x: np.ndarray = np.where(l > (3 * ((6 / 29) ** 3)), ((l + 16) / 116 + a / 500) ** 3, (l / (29 * ((6 / 29) ** 2)) + a / 500) ** 3)
	x = np.where(x > ((6 / 29) ** 3), x, (x - 16 / 116) / 3)

	# Z calculation
	z: np.ndarray = np.where(l > (3 * ((6 / 29) ** 3)), ((l + 16) / 116 - b / 200) ** 3, (l / (29 * ((6 / 29) ** 2)) - b / 200) ** 3)
	z = np.where(z > ((6 / 29) ** 3), z, (z - 16 / 116) / 3)

	# CIEXYZ image
	xyz: np.ndarray = np.stack((x, y, z), axis=0)

	## CIEXYZ to RGB
	# Normalize the CIEXYZ image
	xyz *= ILLUMINANTS[illuminant]	# Multiply by the illuminant

	# CIEXYZ to RGB
	rgb: np.ndarray = np.dot(xyz.T, XYZ_TO_RGB_MATRIX.T).T

	# Return the RGB image
	return rgb * 255


# RGB to CIELUV
def rgb_to_luv(image: np.ndarray, illuminant: str = 'D65') -> np.ndarray:
	""" Convert an RGB image to CIELUV color space\n
	Args:
		image		(np.ndarray):	RGB image (3D array)
		illuminant	(str):			White point illuminant, either 'D65' or 'D50' (default: 'D65')
	Returns:
		(np.ndarray): CIELUV image (3D array)
	Sources:
		https://en.wikipedia.org/wiki/CIELUV#The_forward_transformation
	"""
	# Normalize the image
	normalized: np.ndarray = image / 255

	# RGB to CIEXYZ
	xyz: np.ndarray = np.dot(normalized.T, RGB_TO_XYZ_MATRIX.T).T

	## CIEXYZ to CIELUV
	# Normalize the CIEXYZ image
	if illuminant not in ILLUMINANTS:
		raise ValueError(f"Unknown illuminant '{illuminant}', must be in {list(ILLUMINANTS.keys())}")
	x, y, z = xyz

	# Luminescence calculation
	y_l: np.ndarray = y / ILLUMINANTS[illuminant][1]
	l: np.ndarray = np.where(y_l <= (6/29) ** 3, (29/3) ** 3 * y_l, 116 * y_l ** (1/3) - 16)

	# u' and v' calculation
	DIVIDER: np.ndarray = (x + 15 * y + 3 * z)
	u_prime: np.ndarray = (4 * x) / DIVIDER
	v_prime: np.ndarray = (9 * y) / DIVIDER

	# u'n and v'n calculation
	DIVIDER_N: np.ndarray = (ILLUMINANTS[illuminant][0] + 15 * ILLUMINANTS[illuminant][1] + 3 * ILLUMINANTS[illuminant][2])
	u_prime_n: np.ndarray = (4 * ILLUMINANTS[illuminant][0]) / DIVIDER_N
	v_prime_n: np.ndarray = (9 * ILLUMINANTS[illuminant][1]) / DIVIDER_N

	# u* and v* calculation
	u: np.ndarray = 13 * l * (u_prime - u_prime_n)
	v: np.ndarray = 13 * l * (v_prime - v_prime_n)

	# Return the CIELUV image
	return np.stack((l, u, v), axis=0)

# CIELUV to RGB
def luv_to_rgb(image: np.ndarray, illuminant: str = 'D65') -> np.ndarray:
	""" Convert an CIELUV image to RGB color space\n
	Args:
		image		(np.ndarray):	CIELUV image (3D array)
		illuminant	(str):			White point illuminant, either 'D65' or 'D50' (default: 'D65')
	Returns:
		(np.ndarray): RGB image (3D array)
	Sources:
		https://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation
	"""
	# Unpack the image
	l, u, v = image

	## CIELUV to CIEXYZ
	# u'n and v'n calculation
	DIVIDER_N: np.ndarray = (ILLUMINANTS[illuminant][0] + 15 * ILLUMINANTS[illuminant][1] + 3 * ILLUMINANTS[illuminant][2])
	u_prime_n: np.ndarray = (4 * ILLUMINANTS[illuminant][0]) / DIVIDER_N
	v_prime_n: np.ndarray = (9 * ILLUMINANTS[illuminant][1]) / DIVIDER_N

	# u' and v' calculation
	u_prime: np.ndarray = u / (13 * l) + u_prime_n
	v_prime: np.ndarray = v / (13 * l) + v_prime_n

	# Luminescence calculation
	y: np.ndarray = np.where(l <= 8, ILLUMINANTS[illuminant][1] * l * (3/29) ** 3, ((l + 16) / 116) ** 3)

	# x and z calculation
	x: np.ndarray = y * (9 * u_prime) / (4 * v_prime)
	z: np.ndarray = y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)

	# CIEXYZ image
	xyz: np.ndarray = np.stack((x, y, z), axis=0)

	## CIEXYZ to RGB
	rgb: np.ndarray = np.dot(xyz.T, XYZ_TO_RGB_MATRIX.T).T

	# Return the RGB image
	return rgb * 255

