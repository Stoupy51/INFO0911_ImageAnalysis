
# Import config from the parent folder
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
from print import *

# Imports
import numpy as np
from color_space.all import COLOR_SPACES_CALLS, img_to_sliced_rgb
from descriptors import DESCRIPTORS_CALLS, histogram_hue_per_saturation
from distances import DISTANCES_CALLS
from PIL import Image
from multiprocessing import Pool, cpu_count
from typing import Callable

# Constants
DO_MULTI_PROCESSING: bool = cpu_count() > 4	# Use multiprocessing if more than 4 cores
#DO_MULTI_PROCESSING = False

# Utility function to get clean cache filepath
ALPHANUM = "abcdefghijklmnopqrstuvwxyz0123456789_/"
def clean_cache_path(image_path: str, **kwargs: dict) -> str:
	""" Get the clean cache filepath\n
	Args:
		image_path	(str):	Image path
		kwargs		(dict):		Arguments for the color space and descriptor
	Returns:
		str: Clean cache filepath
	"""
	# Get the color space and descriptor names
	color_space: str = (kwargs.get("color_space", "RGB") + str(kwargs.get("color_space_args", ""))).replace('/', '_')
	descriptor: str = (kwargs.get("descriptor", "") + str(kwargs.get("descriptor_args", ""))).replace('/', '_')

	# Clean the image path and return the cache filepath
	image_name = image_path.replace("\\","/").split("/")[-1].split(".")[0]
	if descriptor:
		return f"{DATABASE_FOLDER}/cache/" + "".join(c for c in f"{image_name}_{color_space}_{descriptor}".lower() if c in ALPHANUM) + ".npz"
	else:
		return f"{DATABASE_FOLDER}/cache/" + "".join(c for c in f"{image_name}_{color_space}".lower() if c in ALPHANUM) + ".npz"


# Function to apply the color space, descriptor, and compute the distance (optional)
def thread_function(image_path: str, color_space: str, descriptor: str, distance: str, color_space_args: dict, descriptor_args: dict, to_compare: np.ndarray|None = None, verbose: bool = False) -> tuple[str, float]:
	""" Thread function to apply the color space, descriptor, and compute the distance (optional)\n
	Args:
		image_path		(str):			Image to process
		color_space		(str):			Color space to use for the descriptor
		descriptor		(str):			Descriptor to use for the search
		distance		(str):			Distance to use for the search
		color_space_args(dict):			Arguments for the color space
		descriptor_args	(dict):			Arguments for the descriptor
		to_compare		(np.ndarray):	Image to compare with (optional, must be pre-processed)
		verbose			(bool):			Verbose mode, default is False
	Returns:
		tuple: Tuple with the following format:
			str:			Path of the image
			float:			Distance between the image and the request
	"""
	image_path = image_path.replace("\\","/")
	descriptor_function: Callable = DESCRIPTORS_CALLS[descriptor]["function"]
	try:
		# Get cache paths
		cache_color_space: str = clean_cache_path(image_path, color_space=color_space, color_space_args=color_space_args)
		cache_descriptor: str = clean_cache_path(image_path, color_space=color_space, color_space_args=color_space_args, descriptor=descriptor, descriptor_args=descriptor_args)

		# Additional descriptor args (if any)
		more_desc_args: dict = {}
		if descriptor == "Histogram":
			if color_space in ["YUV", "YIQ"]:
				more_desc_args["ranges"] = [(0, 256, 1), (0, 1, 0.1), (0, 1, 0.1)]
			elif color_space in ["HSV", "HSL"]:
				more_desc_args["ranges"] = [(0, 360, 1), (0, 1, 0.1), (0, 1, 0.1)]
			elif color_space == "CMYK":
				more_desc_args["ranges"] = [(0, 1, 0.1)] * 4
			elif color_space == "L*a*b":
				more_desc_args["ranges"] = [(0, 100, 1), (-128, 128, 1), (-128, 128, 1)]
			elif color_space == "L*u*v":
				more_desc_args["ranges"] = [(0, 100, 1), (-134, 220, 1), (-140, 122, 1)]

		# Apply the color space and descriptor to the image
		if os.path.exists(cache_descriptor):

			# Load the descriptor from the cache
			image: np.ndarray = np.load(cache_descriptor, allow_pickle=False)["arr_0"]
		else:
			# Try to load the color space applied from the cache
			if os.path.exists(cache_color_space):
				image: np.ndarray = np.load(cache_color_space, allow_pickle=False)["arr_0"]
			else:
				original_image: np.ndarray = np.array(Image.open(image_path).convert("RGB"))
				image: np.ndarray = img_to_sliced_rgb(original_image)
				image = COLOR_SPACES_CALLS[color_space]["function"](image, **color_space_args)
				if type(image) == tuple:
					image = image[0]
				
				# Save the color space applied to the cache
				if color_space != "RGB":
					os.makedirs(os.path.dirname(cache_color_space), exist_ok=True)
					if not os.path.exists(cache_color_space):
						np.savez_compressed(cache_color_space, image)

			image = descriptor_function(image, **descriptor_args, **more_desc_args)

			# Save the descriptor applied to the cache
			os.makedirs(os.path.dirname(cache_descriptor), exist_ok=True)
			if not os.path.exists(cache_descriptor):
				np.savez_compressed(cache_descriptor, image)

		# Compute the distance between the images
		distance_value: float = 0.0
		if to_compare is not None:
			distance_value = DISTANCES_CALLS[distance](image, to_compare)

		# Return the path, image and distance
		if verbose:
			debug(f"Computed distance for '{image_path}' with value {distance_value}")
		return image_path, distance_value
	except KeyboardInterrupt:
		raise KeyboardInterrupt
	except Exception as e:
		error(f"Error while processing '{image_path}' with {color_space} and {descriptor}:\n{e}", exit=False)

# Search engine
def search(image_request: np.ndarray, color_space: str, descriptor: str, distance: str, max_results: int = 10) -> list[tuple[str, np.ndarray, float]]:
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
	cs: dict = COLOR_SPACES_CALLS[color_space]
	cs_args: dict = cs.get("args", {})
	ds: dict = DESCRIPTORS_CALLS[descriptor]
	ds_args: dict = ds.get("args", {})
	image_request = img_to_sliced_rgb(image_request)
	image_request = cs["function"](image_request, **cs_args)
	if type(image_request) == tuple:
		image_request = image_request[0]
	image_request = ds["function"](image_request, **ds_args)

	# Apply the color space and descriptor to the images, then compute the distance
	images_paths: list[str] = [f"{root}/{file}" for root, _, files in os.walk(DATABASE_FOLDER) for file in files if file.endswith(IMAGE_EXTENSIONS)]
	thread_args: list[tuple] = [(image_path, color_space, descriptor, distance, cs_args, ds_args, image_request) for image_path in images_paths]
	if DO_MULTI_PROCESSING:
		with Pool(cpu_count()) as pool:
			debug(f"Using {cpu_count()} processes for the search engine")
			results: list[tuple[str, float]] = pool.starmap(thread_function, thread_args)
			debug(f"Computed {len(results)} images distances")
	else:
		results: list[tuple[str, float]] = [thread_function(*args) for args in thread_args]

	# Sort the images by distance and return the most similar ones
	sorted_images: list[tuple[str, float]] = sorted(results, key=lambda x: x[-1])[:max_results]
	return [(image_path, np.array(Image.open(image_path).convert("RGB")), distance) for image_path, distance in sorted_images]


# Function to compute every single cache
def offline_cache_compute() -> None:
	""" Compute every single cache for the database """
	# Apply the color space and descriptor to the images
	images_paths: list[str] = [f"{root}/{file}" for root, _, files in os.walk(DATABASE_FOLDER) for file in files if file.endswith(IMAGE_EXTENSIONS)]

	# Prepare arguments
	thread_args: list[tuple] = []
	for image_path in images_paths:
		for color_space in COLOR_SPACES_CALLS:
			for descriptor in DESCRIPTORS_CALLS:
				cs: dict = COLOR_SPACES_CALLS[color_space]
				cs_args: dict = cs.get("args", {})
				ds: dict = DESCRIPTORS_CALLS[descriptor]
				ds_args: dict = ds.get("args", {})
				thread_args.append((image_path, color_space, descriptor, "", cs_args, ds_args))
	
	# Compute the cache (using multiprocessing if available)
	if DO_MULTI_PROCESSING:
		with Pool(cpu_count()) as pool:
			debug(f"Using {cpu_count()} processes to compute {len(thread_args)} images caches")
			pool.starmap(thread_function, thread_args)
	else:
		debug(f"Computing {len(thread_args)} images caches")
		for args in thread_args:
			thread_function(*args, verbose=True)
	debug(f"Computed {len(thread_args)} images caches")

