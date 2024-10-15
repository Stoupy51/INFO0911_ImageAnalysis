
# Imports
import os
import numpy as np
from color_space.all import COLOR_SPACES_CALLS, img_to_sliced_rgb
from descriptors import DESCRIPTORS_CALLS
from distances import DISTANCES_CALLS
from PIL import Image
from multiprocessing import Pool, cpu_count
from print import *

# Import config from the parent folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATABASE_FOLDER, IMAGE_EXTENSIONS

DO_MULTI_PROCESSING: bool = True

# Function to apply the color space, descriptor, and compute the distance (optional)
def thread_function(image_path: np.ndarray, color_space: str, descriptor: str, distance: str, color_space_args: dict, descriptor_args: dict, to_compare: np.ndarray|None = None) -> tuple[str, np.ndarray, float]:
	""" Thread function to apply the color space, descriptor, and compute the distance (optional)\n
	Args:
		image_path		(np.ndarray):	Image to process
		color_space		(str):			Color space to use for the descriptor
		descriptor		(str):			Descriptor to use for the search
		distance		(str):			Distance to use for the search
		color_space_args(dict):			Arguments for the color space
		descriptor_args	(dict):			Arguments for the descriptor
		to_compare		(np.ndarray):	Image to compare with (optional, must be pre-processed)
	Returns:
		tuple: Tuple with the following format:
			str:			Path of the image
			np.ndarray:		Image in the database
			float:			Distance between the image and the request
	"""
	# Apply the color space and descriptor to the image
	original_image: np.ndarray = np.array(Image.open(image_path))
	image = img_to_sliced_rgb(original_image)
	if image.shape[0] != 3:
		print(f"Shape of '{image_path}': {image.shape}")	# TODO: Make sure the image is RGB
		if len(image.shape) == 2:
			image = np.stack([image] * 3)
		elif len(image.shape) == 3 and image.shape[0] == 4:
			image = image[:3]
	image = COLOR_SPACES_CALLS[color_space](image, **color_space_args)
	image = DESCRIPTORS_CALLS[descriptor](image, **descriptor_args)

	# Compute the distance between the images
	distance_value: float = 0.0
	if to_compare is not None:
		distance_value = DISTANCES_CALLS[distance](image, to_compare)

	# Return the path, image and distance
	return image_path, original_image, distance_value

# Search engine
def search(image_request: np.ndarray, color_space: str, descriptor: str, distance: str, max_results: int = 5) -> list[tuple[str, np.ndarray, float]]:
	""" Search for similar images in the database\n
	Args:
		image_request	(np.ndarray):	Image to search for, example shape: (100, 100, 3)
		color_space		(str):			Color space to use for the descriptor
		descriptor		(str):			Descriptor to use for the search
		distance		(str):			Distance to use for the search
		max_results		(int):			Maximum number of results to return, default is 5
	Returns:
		list[tuple]: List of tuples with the following format:
			str:			Path of the image
			np.ndarray:		Image in the database (original)
			float:			Distance between the image and the request
	"""
	# Check if the color space, descriptor and distance are valid
	assert color_space in COLOR_SPACES_CALLS, f"Color space '{color_space}' not found in {list(COLOR_SPACES_CALLS.keys())}"
	assert descriptor in DESCRIPTORS_CALLS, f"Descriptor '{descriptor}' not found in {list(DESCRIPTORS_CALLS.keys())}"
	assert distance in DISTANCES_CALLS, f"Distance '{distance}' not found in {list(DISTANCES_CALLS.keys())}"

	# Apply the color space and descriptor to the request image
	color_space_args: dict = {}	# TODO: Possible to add arguments in the future for the color space
	descriptor_args: dict = {}	# TODO: Possible to add arguments in the future for the descriptor
	image_request = img_to_sliced_rgb(image_request)
	image_request = COLOR_SPACES_CALLS[color_space](image_request, **color_space_args)
	image_request = DESCRIPTORS_CALLS[descriptor](image_request, **descriptor_args)

	# Apply the color space and descriptor to the images, then compute the distance
	images_paths: list[str] = [f"{root}/{file}" for root, _, files in os.walk(DATABASE_FOLDER) for file in files if file.endswith(IMAGE_EXTENSIONS)]
	thread_args: list[tuple] = [(image_path, color_space, descriptor, distance, color_space_args, descriptor_args, image_request) for image_path in images_paths]
	if DO_MULTI_PROCESSING:
		with Pool(cpu_count()) as pool:
			debug(f"Using {cpu_count()} processes for the search engine")
			results: list[tuple[str, np.ndarray, float]] = pool.starmap(thread_function, thread_args)
			debug(f"Computed {len(results)} images distances")
	else:
		results: list[tuple[str, np.ndarray, float]] = [thread_function(*args) for args in thread_args]

	# Sort the images by distance and return the most similar ones
	return sorted(results, key=lambda x: x[2])[:max_results]

