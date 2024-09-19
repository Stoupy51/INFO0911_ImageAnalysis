
## Imports
from fastaniso import anisodiff
from PIL import Image
import numpy as np
import os

# Constants
IMAGE_PATH: str = "Peppers.png"
IMAGE_NAME: str = IMAGE_PATH.split(".")[0]
NB_ITERATIONS: list[int] =	[15, 20, 50]				# Nombre d'itérations
KAPPA: list[int] = 			[15, 50, 100]				# Seuil de contraste
GAMMA: list[float] =		[0.1, 0.25, 0.5, 1.0]		# Coefficient de conduction
FORMULA: list[int] =		[1,2]						# 1 utilise exp, 2 utilise la fonction inverse

NOISES: list[str] = [
	"speckle_0_200",
	"speckle_0_80",
	"speckle_0_50",
	"none"
]

# Load image and convert to grayscale if not grayscale
BASE_IMAGE: np.ndarray = np.array(Image.open(IMAGE_PATH))
if BASE_IMAGE.ndim == 3:
	BASE_IMAGE = BASE_IMAGE.mean(2)

if __name__ == "__main__":

	# For each noise,
	for noise in NOISES:

		# Make folder
		os.makedirs(IMAGE_NAME, exist_ok=True)
		os.makedirs(f"{IMAGE_NAME}/{noise}", exist_ok=True)

		# Apply noise to the image
		if "speckle" in noise:
			params: list[str] = noise.split("_")
			mean: float = int(params[1]) / 100
			stddev: float = int(params[2]) / 100

			random_matrix: np.ndarray = np.random.normal(mean, stddev, size=BASE_IMAGE.shape)
			noised_image: np.ndarray = BASE_IMAGE + BASE_IMAGE.std() * random_matrix
		elif "none" == noise:
			noised_image: np.ndarray = BASE_IMAGE
		
		# Save the initial noisy image
		noised_image_path: str = f"{IMAGE_NAME}/{noise}/_{noise}.png"
		noised_image_image: Image.Image = Image.fromarray(noised_image).convert("L")
		noised_image_image.save(noised_image_path)

		# For each parameter,
		for nb_iter in NB_ITERATIONS:
			iter_str: str = f"_iter{nb_iter}"
			for k in KAPPA:
				kappa_str: str = f"_kappa{k}"
				for gamma in GAMMA:
					gamma_str: str = f"_gamma{gamma}"
					for formula in FORMULA:
						formula_str: str = "exponentielle" if formula == 1 else "inverse"

						# Apply anisotropic diffusion
						output: np.ndarray = anisodiff(noised_image, niter=nb_iter, kappa=k, gamma=gamma, option=formula, ploton=False)

						# Save image
						new_image: Image.Image = Image.fromarray(output).convert("L")
						save_path: str = f"{IMAGE_NAME}/{noise}/{formula_str}{iter_str}{kappa_str}{gamma_str}.png"
						new_image.save(save_path)

