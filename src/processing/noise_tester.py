
## Imports
from fastaniso import anisodiff
from typing import Iterable
from PIL import Image
import numpy as np
import os

def noise_tester(output_directory: str, flatten_image: np.ndarray, noise: str, nb_iterations: Iterable[int], kappa: Iterable[int], gamma: Iterable[float], formulas: Iterable[int], color_palette: str = "L") -> None:
	""" Apply anisotropic diffusion to a generated noisy image and saves the results in a specific folder.\n
	Args:
		saving_directory		(str):				The directory where the images will be saved.
		flatten_image			(np.ndarray):		The image to apply the anisotropic diffusion to.
		noise					(str):				The noise to apply to the image, formatted as "noisename_p1_p2_..._pN" where pX is divided by 100, e.g. "speckle_0_200" for a speckle noise with mean 0 and stddev 2.
		nb_iterations			(Iterable[int]):	The number of iterations to apply, e.g. [15, 20, 50].
		kappa					(Iterable[int]):	The contrast threshold, e.g. [15, 50, 100].
		gamma					(Iterable[float]):	The conduction coefficient, e.g. [0.1, 0.25, 0.5, 1.0].
		formulas				(Iterable[int]):	The formulas to use (1 for exponential, 2 for inverse), e.g. [1, 2].
		color_palette			(str):				The color palette to use, e.g. "L" for grayscale, "RGB" for color.
	Returns:
		None
	"""
	# Make folders
	saving_directory: str = f"{output_directory}/{noise}"
	os.makedirs(saving_directory, exist_ok=True)

	# Apply noise to the image
	if "speckle" in noise:
		params: list[str] = noise.split("_")
		mean: float = int(params[1]) / 100
		stddev: float = int(params[2]) / 100

		random_matrix: np.ndarray = np.random.normal(mean, stddev, size=flatten_image.shape)
		noised_image: np.ndarray = flatten_image + flatten_image.std() * random_matrix
	elif "none" == noise:
		noised_image: np.ndarray = flatten_image
	
	# Save the initial noisy image with 95% quality
	noised_image_path: str = f"{saving_directory}/_{noise}.jpg"
	noised_image_image: Image.Image = Image.fromarray(noised_image).convert(color_palette)
	noised_image_image.save(noised_image_path, quality = 95)

	# For each parameter,
	for nb_iter in nb_iterations:
		iter_str: str = f"iter{nb_iter}"
		for k in kappa:
			kappa_str: str = f"k{k}"
			for g in gamma:
				gamma_str: str = f"g{g}"
				for formula in formulas:
					formula_str: str = "exponentielle" if formula == 1 else "inverse"

					# Apply anisotropic diffusion
					output: np.ndarray = anisodiff(noised_image, niter=nb_iter, kappa=k, gamma=g, option=formula, ploton=False)

					# Save image
					new_image: Image.Image = Image.fromarray(output).convert("L")
					save_path: str = f"{saving_directory}/{formula_str}_{iter_str}_{kappa_str}_{gamma_str}.jpg"
					new_image.save(save_path, quality = 95)

