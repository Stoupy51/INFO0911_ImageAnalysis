
# RGB to Grayscale (average, norm 709, norm 601)
# RGB to YIQ, YUV, I1I2I3, RGB Normalisé	(combinaisons linéaires)
# RGB to HSL, HSV, CMYK, L*a*b, L*u*v		(combinaisons non-linéaires)
# Indexation single and multi channels

# Import every function from the other files
from src.color_space.common import *
from src.color_space.utils import *
from src.color_space.grayscale import *
from src.color_space.linears import *
from src.color_space.non_linears import *
from src.color_space.indexation import *

# Name every function
from typing import Callable
COLOR_SPACES_CALLS: dict[str, Callable] = {
	# Grayscales
	"Grayscale (average)":			rgb_to_grayscale_average,
	"Grayscale (norm 709)":			rgb_to_grayscale_norm709,
	"Grayscale (norm 601)":			rgb_to_grayscale_norm601,

	# Linears
	"YIQ":							rgb_to_yiq,
	"YUV":							rgb_to_yuv,
	"I1I2I3":						rgb_to_i1i2i3,
	"RGB Normalisé":				rgb_to_rgb_normalized,

	# Non-linears
	"HSL":							rgb_to_hsl,
	"HSV":							rgb_to_hsv,
	"CMYK":							rgb_to_cmyk,
	"L*a*b":						rgb_to_lab,
	"L*u*v":						rgb_to_luv,

	# Indexation
	"Indexation single channel":	indexed_single_channel,
	"Indexation multi channels":	indexed_multi_channels,
}

