
# Imports
from src.distances import *
from PIL import Image
import numpy as np
import os

# measure_all_distances(output_dir, flatten_original, noise, color_palette=color_palette)
def measure_all_distances(output_dir: str, flatten_original: np.ndarray, noise: str, color_palette: str = "L") -> None:
	""" Measure all distances between the original image and the images in the output directory\n
	And generates txt files with the results in the same output directory\n
	Args:
		output_dir			(str):			The output directory
		flatten_original	(np.ndarray):	The original image
		noise				(str):			The noise applied to the images
		color_palette		(str):			The color palette to use, e.g. "L" for grayscale, "RGB" for color
	Returns:
		None
	"""
	# Get all generated images
	images: list[str] = [f for f in os.listdir(f"{output_dir}/{noise}") if f.endswith((".jpg",".png"))]
	distance_per_image_per_distance: dict[str, dict[str, float]] = {}

	# For each image,
	for image_path in images:

		# Load image and convert as a flatten np array
		image: np.ndarray = np.array(Image.open(f"{output_dir}/{noise}/{image_path}").convert(color_palette)).flatten()

		# Calculate distance for each distance function
		distance_per_image_per_distance[image_path] = {}
		for distance_name, distance_function in DISTANCES_CALLS.items():
			distance_per_image_per_distance[image_path][distance_name] = distance_function(flatten_original, image)
	
	# Write the results in a txt file
	for distance_name in DISTANCES_CALLS.keys():

		# Get sorted images by distance
		images = sorted(images, key=lambda x: distance_per_image_per_distance[x][distance_name])

		# Get the content to write
		to_write: str = "\n"
		for image_path in images:
			to_write += image_path + "\t" + str(distance_per_image_per_distance[image_path][distance_name]) + "\n"

		# Write the content in a txt file
		with open(f"{output_dir}/{noise}/distances_{distance_name}.txt", "w") as f:
			f.write(to_write + "\n")

