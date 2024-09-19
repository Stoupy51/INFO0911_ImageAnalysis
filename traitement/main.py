
## Imports
from src.processing.noise_tester import noise_tester
from PIL import Image
import numpy as np


# Constants
IMAGE_PATH: str = "Peppers.png"
IMAGE_NAME: str = IMAGE_PATH.split(".")[0]
NB_ITERATIONS: list[int] =	[15, 20, 50]				# Nombre d'itérations
KAPPA: list[int] = 			[15, 50, 100]				# Seuil de contraste
GAMMA: list[float] =		[0.1, 0.25, 0.5, 1.0]		# Coefficient de conduction
FORMULA: list[int] =		[1,2]						# 1 utilise exp, 2 utilise la fonction inverse

NOISES: list[str] = [
	"speckle_0_200",	# moyenne 0, écart-type 200
	"speckle_0_80",		# moyenne 0, écart-type 80
	"speckle_0_50",		# moyenne 0, écart-type 50

	"none"				# Pas de bruit
]

# Load image and convert to grayscale if not grayscale
BASE_IMAGE: np.ndarray = np.array(Image.open(IMAGE_PATH))
if BASE_IMAGE.ndim == 3:
	BASE_IMAGE = BASE_IMAGE.mean(2)

if __name__ == "__main__":

	# For each noise,
	for noise in NOISES:
		noise_tester(IMAGE_NAME, BASE_IMAGE, noise, NB_ITERATIONS, KAPPA, GAMMA, FORMULA, color_palette="L")

