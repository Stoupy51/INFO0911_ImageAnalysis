
## Imports
from config import *
from src.processing.noise_tester import noise_tester
from src.processing.distance_tests import measure_all_distances
from src.distances import DISTANCES_CALLS
from PIL import Image
from typing import Iterable
import numpy as np

# Function that will make extensive tests
def extensive_tests(noises: Iterable[str], nb_iterations: Iterable[int], kappa: Iterable[int], gamma: Iterable[float], formulas: Iterable[int], color_palette: str = "L") -> None:
	""" Apply anisotropic diffusion to a generated noisy image and saves the results in a specific folder.\n
	Args:
		noises					(Iterable[str]):	The noises to apply to the image, e.g. ["speckle_0_200", "speckle_0_80", "speckle_0_50", "none"].
		nb_iterations			(Iterable[int]):	The number of iterations to apply, e.g. [15, 20, 50].
		kappa					(Iterable[int]):	The contrast threshold, e.g. [15, 50, 100].
		gamma					(Iterable[float]):	The conduction coefficient, e.g. [0.1, 0.25, 0.5, 1.0].
		formulas				(Iterable[int]):	The formulas to use (1 for exponential, 2 for inverse), e.g. [1, 2].
		color_palette			(str):				The color palette to use, e.g. "L" for grayscale, "RGB" for color.
	Returns:
		None
	"""
	# Get all the images and make the output folder
	images: list[str] = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg",".png"))]
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)

	# For each image,
	for image in images:
		
		# Get the output directory for the image
		output_dir: str = f"{OUTPUT_FOLDER}/{image.split('.')[0]}"

		# If all .txt noises files are already present, ask the user if he wants to skip the image
		if all([os.path.exists(f"{output_dir}/{noise}/distances_{distance}.txt") for noise in noises for distance in DISTANCES_CALLS.keys()]):
			if input(f"Image {image} already processed, do you want to skip it? (Y/n): ").lower() != "n":
				continue

		# Load image and convert to grayscale if not grayscale, and flatten it
		base_image: np.ndarray = np.array(Image.open(f"{IMAGE_FOLDER}/{image}"))
		if base_image.ndim == 3:
			base_image = base_image.mean(2)
		flatten_original: np.ndarray = base_image.flatten()

		# For each noise, launch the noise tester
		for noise in noises:
			noise_tester(output_dir, flatten_original, base_image.shape, noise, nb_iterations, kappa, gamma, formulas, color_palette=color_palette)
			measure_all_distances(output_dir, flatten_original, noise, color_palette=color_palette)

