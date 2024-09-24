
# RGB to Grayscale (average, norm 709, norm 601)
# RGB to YIQ, YUV, I1I2I3, RGB Normalisé	(combinaisons linéaires)
# RGB to HSL, HSV, CMYK, L*a*b, L*u*v		(combinaisons non-linéaires)

# Import every function from the other files
from src.color_space.common import *
from src.color_space.utils import *
from src.color_space.grayscale import *
from src.color_space.linears import *
from src.color_space.non_linears import *

