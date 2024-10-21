
# RGB to Grayscale (average, norm 709, norm 601)
# RGB to YIQ, YUV, I1I2I3, RGB Normalisé	(combinaisons linéaires)
# RGB to HSL, HSV, CMYK, L*a*b, L*u*v		(combinaisons non-linéaires)
# Indexation single and multi channels

# Import every function from the other files
from .common import *
from .utils import *
from .grayscale import *
from .linears import *
from .non_linears import *
from .indexation import *

# Name every function
from typing import Callable
COLOR_SPACES_CALLS: dict[str, Callable] = {
	# Classic
	"RGB":									{"function":rgb_to_rgb, "args":{}},

	# Grayscales
	"Grayscale (average)":					{"function":rgb_to_grayscale_average, "args":{}},
	"Grayscale (norm 709)":					{"function":rgb_to_grayscale_norm709, "args":{}},
	"Grayscale (norm 601)":					{"function":rgb_to_grayscale_norm601, "args":{}},

	# Linears
	"YIQ":									{"function":rgb_to_yiq, "args":{}},
	"YUV":									{"function":rgb_to_yuv, "args":{}},
	"I1I2I3":								{"function":rgb_to_i1i2i3, "args":{}},
	"RGB Normalisé":						{"function":rgb_to_rgb_normalized, "args":{}},

	# Non-linears
	"HSL":									{"function":rgb_to_hsl, "args":{}},
	"HSV":									{"function":rgb_to_hsv, "args":{}},
	"CMYK":									{"function":rgb_to_cmyk, "args":{}},
	"L*a*b":								{"function":rgb_to_lab, "args":{}},
	"L*u*v":								{"function":rgb_to_luv, "args":{}},

	# Indexation
	"Indexation (2,2,2)":	{"function":indexed_multi_channels, "args":{"levels":[2, 2, 2]}},
	"Indexation (4,4,4)":	{"function":indexed_multi_channels, "args":{"levels":[4, 4, 4]}},
	"Indexation (8,8,8)":	{"function":indexed_multi_channels, "args":{"levels":[8, 8, 8]}},
}

