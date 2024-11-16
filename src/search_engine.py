
# Import config from the parent folder
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
from print import *

# Imports
import numpy as np
from src.color_space.all import COLOR_SPACES_CALLS, img_to_sliced_rgb
from src.descriptors import DESCRIPTORS_CALLS
from src.distances import DISTANCES_CALLS
from src.image import ImageData
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
		image_path	(str):		Image path
		kwargs		(dict):		Arguments for the color space and descriptor
	Returns:
		str: Clean cache filepath
	"""
	# Get the color space and descriptor names
	color_space: str = (kwargs.get("color_space", "RGB") + str(kwargs.get("color_space_args", ""))).replace('/', '_')
	descriptors: str = ("_".join(kwargs.get("descriptors", []))).replace('/', '_')

	# Clean the image path and return the cache filepath
	image_name = image_path.replace("\\","/").split("/")[-1].split(".")[0]
	if descriptors:
		return f"{DATABASE_FOLDER}/cache/" + "".join(c for c in f"{image_name}_{color_space}_{descriptors}".lower() if c in ALPHANUM) + ".npz"
	else:
		return f"{DATABASE_FOLDER}/cache/" + "".join(c for c in f"{image_name}_{color_space}".lower() if c in ALPHANUM) + ".npz"

# Function to resize the image down to the maximum size
def resize_down(image: Image.Image, max_size: tuple[int, int] = SEARCH_MAX_IMAGE_SIZE, min_or_max: Callable = max) -> Image.Image:
	""" Resize the image down to the maximum size while preserving aspect ratio\n
	Args:
		image			(Image.Image):		Image to resize
		max_size		(tuple[int, int]):	Maximum size to resize to
		min_or_max		(Callable):			Function to use to get the minimum or maximum of the two ratios
	Returns:
		Image.Image: Resized image
	"""
	width, height = image.size
	if width > max_size[0] or height > max_size[1]:
		# Calculate scaling factor to fit within max dimensions while preserving ratio
		scale_w = max_size[0] / width
		scale_h = max_size[1] / height
		scale = min_or_max(scale_w, scale_h)
		
		# Calculate new dimensions
		new_width = int(width * scale)
		new_height = int(height * scale)
		image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
	return image

# Function to apply the color space, descriptor, and compute the distance (optional)
def thread_function(image_path: str, color_space: str, descriptors: list[str], distance: str, color_space_args: dict, to_compare: np.ndarray|None = None, verbose: bool = False) -> tuple[str, float]:
	""" Thread function to apply the color space, descriptor, and compute the distance (optional)\n
	Args:
		image_path		(str):			Image to process
		color_space		(str):			Color space to use for the descriptor
		descriptors		(list[str]):	List of descriptors to use in order
		distance		(str):			Distance to use for the search
		color_space_args(dict):			Arguments for the color space
		to_compare		(np.ndarray):	Image to compare with (optional, must be pre-processed)
		verbose			(bool):			Verbose mode, default is False
	Returns:
		tuple: Tuple with the following format:
			str:			Path of the image
			float:			Distance between the image and the request
	"""
	@handle_error(message=f"Error while processing '{image_path}' with {color_space} and {descriptors}", error_log=2)
	def intern():
		# Get cache paths
		cleaned_path = image_path.replace("\\","/")
		cache_color_space: str = clean_cache_path(cleaned_path, color_space=color_space, color_space_args=color_space_args)
		cache_descriptor: str = clean_cache_path(cleaned_path, color_space=color_space, color_space_args=color_space_args, descriptors=descriptors)

		# Apply the color space and descriptors to the image
		if os.path.exists(cache_descriptor):

			# Load the descriptor from the cache
			data: np.ndarray = np.load(cache_descriptor, allow_pickle=False)["arr_0"]
			img: ImageData = ImageData(data, "Descriptor")
		else:
			# Try to load the color space applied from the cache
			if os.path.exists(cache_color_space):
				data: np.ndarray = np.load(cache_color_space, allow_pickle=False)["arr_0"]
				img: ImageData = ImageData(data, color_space)
			else:
				original_image: Image.Image = Image.open(cleaned_path).convert("RGB")
				original_image = resize_down(original_image)
				data: np.ndarray = img_to_sliced_rgb(np.array(original_image))
				img: ImageData = ImageData(data, "RGB")
				img = COLOR_SPACES_CALLS[color_space]["function"](img, **color_space_args)
				
				# Save the color space applied to the cache
				if color_space != "RGB":
					os.makedirs(os.path.dirname(cache_color_space), exist_ok=True)
					if not os.path.exists(cache_color_space):
						np.savez_compressed(cache_color_space, img.data)

			# Apply each descriptor in sequence (if the descriptor is not compatible with the color space, return None)
			for descriptor in descriptors:
				if "HSV/HSL" in descriptor and img.color_space not in ["HSV", "HSL"]:
					return None
				ds: dict = DESCRIPTORS_CALLS[descriptor]
				ds_args: dict = ds.get("args", {})
				img = ds["function"](img, **ds_args)

			# Save the descriptor applied to the cache
			os.makedirs(os.path.dirname(cache_descriptor), exist_ok=True)
			if not os.path.exists(cache_descriptor):
				np.savez_compressed(cache_descriptor, img.data)

		# Compute the distance between the images
		distance_value: float = 0.0
		if distance is not None and to_compare is not None:
			distance_value = DISTANCES_CALLS[distance](img.data, to_compare)

		# Return the path, image and distance
		if verbose:
			debug(f"Computed distance for '{cleaned_path}' with value {distance_value}")
		return cleaned_path, distance_value
	return intern()

# Search engine
@handle_error(message="Error during the search engine")
def search(image_request: np.ndarray, color_space: str, descriptors: list[str], distance: str, max_results: int = 10) -> list[tuple[str, np.ndarray, float]]:
	""" Search for similar images in the database\n
	Args:
		image_request	(np.ndarray):	Image to search for, example shape: (100, 100, 3)
		color_space		(str):			Color space to use for the descriptor
		descriptors		(list[str]):	List of descriptors to use in order
		distance		(str):			Distance to use for the search
		max_results		(int):			Maximum number of results to return, default is 5
	Returns:
		list[tuple]: List of tuples with the following format:
			str:			Path of the image
			np.ndarray:		Image in the database (original)
			float:			Distance between the image and the request
	"""
	# Check if the color space, descriptors and distance are valid
	assert color_space in COLOR_SPACES_CALLS, f"Color space '{color_space}' not found in {list(COLOR_SPACES_CALLS.keys())}"
	for descriptor in descriptors:
		assert descriptor in DESCRIPTORS_CALLS, f"Descriptor '{descriptor}' not found in {list(DESCRIPTORS_CALLS.keys())}"
	assert distance in DISTANCES_CALLS, f"Distance '{distance}' not found in {list(DISTANCES_CALLS.keys())}"

	# Apply the color space and descriptors to the request image
	cs: dict = COLOR_SPACES_CALLS[color_space]
	cs_args: dict = cs.get("args", {})
	
	# Start with color space conversion
	image_request = img_to_sliced_rgb(image_request)
	img: ImageData = ImageData(image_request, "RGB")
	img = cs["function"](img, **cs_args)
	
	# Apply each descriptor in sequence
	for descriptor in descriptors:
		ds: dict = DESCRIPTORS_CALLS[descriptor]
		ds_args: dict = ds.get("args", {})
		img = ds["function"](img, **ds_args)

	# Apply the color space and descriptors to the images, then compute the distance
	images_paths: list[str] = [f"{root}/{file}" for root, _, files in os.walk(DATABASE_FOLDER) for file in files if file.endswith(IMAGE_EXTENSIONS)]
	thread_args: list[tuple] = [(image_path, color_space, descriptors, distance, cs_args, img.data) for image_path in images_paths]
	if DO_MULTI_PROCESSING:
		with Pool(cpu_count()) as pool:
			debug(f"Using {cpu_count()} processes for the search engine")
			results: list[tuple[str, float]] = pool.starmap(thread_function, thread_args)
			debug(f"Computed {len(results)} images distances")
	else:
		results: list[tuple[str, float]] = [thread_function(*args) for args in thread_args]
	
	# Filter None results
	results = [result for result in results if result is not None]

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
				thread_args.append((image_path, color_space, [descriptor], None, cs_args))
	
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

