
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
from src.normalization import NORMALIZATION_CALLS
from src.image import ImageData
from PIL import Image
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal
import pickle

# Constants
DO_MULTI_PROCESSING: bool = cpu_count() > 4	# Use multiprocessing if more than 4 cores
PARALLEL_TYPE = Literal["none", "thread", "process"]
#DO_MULTI_PROCESSING = False

# Utility function to get clean cache filepath
ALPHANUM = "abcdefghijklmnopqrstuvwxyz0123456789_/"
def clean_cache_path(image_path: str, **kwargs: dict) -> str:
	""" Get the clean cache filepath\n
	Args:
		image_path	(str):		Image path
		kwargs		(dict):		Arguments for the color spaces and descriptors
	Returns:
		str: Clean cache filepath
	"""
	# Get the color spaces, descriptor and normalization names
	color_spaces: str = ("_".join(kwargs.get("color_spaces", ["RGB"]))).replace('/', '_')
	descriptors: str = ("_".join(kwargs.get("descriptors", []))).replace('/', '_')
	normalization: str = kwargs.get("normalization", "")

	# Clean the image path and return the cache filepath
	image_name = image_path.replace("\\","/").split("/")[-1].split(".")[0]
	if descriptors:
		if normalization:
			return f"{CACHE_FOLDER}/" + "".join(c for c in f"{image_name}_{color_spaces}_{descriptors}_{normalization}".lower() if c in ALPHANUM) + ".npz"
		return f"{CACHE_FOLDER}/" + "".join(c for c in f"{image_name}_{color_spaces}_{descriptors}".lower() if c in ALPHANUM) + ".npz"
	else:
		if normalization:
			return f"{CACHE_FOLDER}/" + "".join(c for c in f"{image_name}_{color_spaces}_{normalization}".lower() if c in ALPHANUM) + ".npz"
		return f"{CACHE_FOLDER}/" + "".join(c for c in f"{image_name}_{color_spaces}".lower() if c in ALPHANUM) + ".npz"

# Function to resize the image down to the maximum size
def resize_down(image: Image.Image, max_size: tuple[int, int] = SEARCH_MAX_IMAGE_SIZE, min_or_max: Callable = max, keep_ratio: bool = False) -> Image.Image:
	""" Resize the image down to the maximum size while preserving aspect ratio\n
	Args:
		image			(Image.Image):		Image to resize
		max_size		(tuple[int, int]):	Maximum size to resize to
		min_or_max		(Callable):			Function to use to get the minimum or maximum of the two ratios
		keep_ratio		(bool):				Keep the aspect ratio, default is False
	Returns:
		Image.Image: Resized image
	"""
	if keep_ratio:
		width, height = image.size
		if width > max_size[0] or height > max_size[1]:
			# Calculate scaling factor to fit within max dimensions while preserving ratio
			scale_w = max_size[0] / width
			scale_h = max_size[1] / height
			scale = min_or_max(scale_w, scale_h)
			
			# Calculate new dimensions
			new_width = int(width * scale)
			new_height = int(height * scale)
			return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
	else:
		width, height = image.size
		return image.resize((max_size[0], max_size[1]), Image.Resampling.LANCZOS)

def load_and_process_image(image_path: str, color_spaces: list[str], descriptors: list[str], normalization: str = "") -> ImageData:
	""" Load and process an image with color spaces and descriptors\n
	Args:
		image_path		(str):			Image to process
		color_spaces	(list[str]):	List of color spaces to use in order
		descriptors		(list[str]):	List of descriptors to use in order
		normalization	(str):			Normalization method to use
	Returns:
		ImageData: Processed image data
	"""
	# Get cache paths
	cleaned_path = image_path.replace("\\","/")
	cache_paths = {
		"color_space": clean_cache_path(cleaned_path, color_spaces=color_spaces),
		"descriptor": clean_cache_path(cleaned_path, color_spaces=color_spaces, descriptors=descriptors),
		"normalized": clean_cache_path(cleaned_path, color_spaces=color_spaces, descriptors=descriptors, normalization=normalization) if normalization else None
	}

	# Try loading from normalized cache first
	if cache_paths["normalized"] and os.path.exists(cache_paths["normalized"]):
		data = np.load(cache_paths["normalized"], allow_pickle=False)["arr_0"]
		return ImageData(data, "Normalized")

	# Try loading from descriptor cache
	if os.path.exists(cache_paths["descriptor"]):
		data = np.load(cache_paths["descriptor"], allow_pickle=False)["arr_0"]
		img = ImageData(data, "Descriptor")
		return apply_normalization(img, normalization, cache_paths["normalized"])

	# Load and process from color space cache or original image
	img: ImageData = load_color_space_data(cleaned_path, cache_paths["color_space"], color_spaces)
	
	# Apply descriptors in parallel and concatenate results
	descriptor_results = []
	for descriptor in descriptors:
		if "HSV/HSL" in descriptor and img.color_space not in ["HSV", "HSL"]:
			return None
		ds = DESCRIPTORS_CALLS[descriptor]
		result: ImageData = ds["function"](img, **ds.get("args", {}))
		descriptor_results.append(result.data)
	
	# Concatenate all descriptor results
	if descriptor_results:
		img = ImageData(np.concatenate(descriptor_results), "Descriptor")

	# Cache descriptor result
	os.makedirs(os.path.dirname(cache_paths["descriptor"]), exist_ok=True)
	if not os.path.exists(cache_paths["descriptor"]):
		np.savez_compressed(cache_paths["descriptor"], img.data)

	return apply_normalization(img, normalization, cache_paths["normalized"])

def load_color_space_data(image_path: str, cache_path: str, color_spaces: list[str]) -> ImageData:
	"""Helper function to load and process color space data"""
	if os.path.exists(cache_path):
		data = np.load(cache_path, allow_pickle=False)["arr_0"]
		return ImageData(data, color_spaces[-1])
	
	original_image = Image.open(image_path).convert("RGB")
	original_image = resize_down(original_image)
	data = img_to_sliced_rgb(np.array(original_image))
	img = ImageData(data, "RGB")

	# Apply color spaces
	for color_space in color_spaces:
		cs = COLOR_SPACES_CALLS[color_space]
		img = cs["function"](img, **cs.get("args", {}))

	# Cache color space result if not RGB
	if color_spaces != ["RGB"]:
		os.makedirs(os.path.dirname(cache_path), exist_ok=True)
		if not os.path.exists(cache_path):
			np.savez_compressed(cache_path, img.data)

	return img

def apply_normalization(img: ImageData, normalization: str, cache_path: str|None) -> ImageData:
	"""Helper function to apply normalization"""
	if not normalization:
		return img
		
	norm = NORMALIZATION_CALLS[normalization]
	img.data = norm["function"](img.data, **norm.get("args", {}))

	if cache_path:
		os.makedirs(os.path.dirname(cache_path), exist_ok=True)
		if not os.path.exists(cache_path):
			np.savez_compressed(cache_path, img.data)

	return img

def thread_function(image_path: str, color_spaces: list[str], descriptors: list[str], normalization: str, distance: str, to_compare: np.ndarray|None = None, verbose: bool = False) -> tuple[str, float]:
	""" Thread function to apply the color spaces, descriptor, and compute the distance (optional)\n
	Args:
		image_path		(str):			Image to process
		color_spaces	(list[str]):	List of color spaces to use in order
		descriptors		(list[str]):	List of descriptors to use in order
		normalization	(str):			Normalization method to use
		distance		(str):			Distance to use for the search
		to_compare		(np.ndarray):	Image to compare with (optional, must be pre-processed)
		verbose			(bool):			Verbose mode, default is False
	Returns:
		str:	Path of the image
		float:	Distance between the image and the request
	"""
	@handle_error(message=f"Error while processing '{image_path}' with {color_spaces} and {descriptors}", error_log=2)
	def intern():
		cleaned_path: str = image_path.replace("\\","/")
		compute_distance: bool = distance is not None and to_compare is not None
		
		img: ImageData|None = load_and_process_image(image_path, color_spaces, descriptors, normalization)
		if img is None:
			return None

		# Compute the distance between the images
		distance_value: float = 0.0
		if compute_distance:
			distance_value = DISTANCES_CALLS[distance](img.data, to_compare)

		# Return the path, image and distance
		if verbose:
			debug(f"Computed distance for '{cleaned_path}' with value {distance_value}")
		return cleaned_path, distance_value
	return intern()

# Search engine
def get_search_cache_path(image_path: str, color_spaces: list[str], descriptors: list[str], normalization: str, distance: str, max_results: int) -> str:
	"""Get the cache path for search results"""
	image_name: str = os.path.splitext(os.path.basename(image_path))[0]
	params: str = f"{image_name}_{'-'.join(color_spaces)}_{'-'.join(descriptors)}_{normalization}_{distance}_{max_results}"
	clean_params: str = "".join(c for c in params.lower() if c in ALPHANUM)
	return f"{CACHE_FOLDER}/search_results/{clean_params}.pkl"


@handle_error(message="Error during the search engine", error_log=2)
def search(
	image_request: Image.Image|str,
	color_spaces: list[str],
	descriptors: list[str], 
	normalization: str,
	distance: str,
	max_results: int = 10,
	parallel: PARALLEL_TYPE = "none"
) -> list[tuple[str, np.ndarray, float]]:
	""" Search for similar images in the database\n
	Args:
		image_request   (Image.Image|str):  Image to search for (str means cache path)
		color_spaces    (list[str]):        List of color spaces to use in order
		descriptors     (list[str]):        List of descriptors to use in order
		normalization   (str):              Normalization method to use
		distance        (str):              Distance to use for the search
		max_results     (int):              Maximum number of results to return, default is 10
		parallel        (str):              Parallelization type: "none", "thread", or "process"
	Returns:
		list[tuple]: List of tuples with the following format:
			str:            Path of the image
			Image.Image:    Image in the database (original)
			float:          Distance between the image and the request
	"""
	# Check if we have cached results for this exact search (only paths and distances)
	if isinstance(image_request, str):
		cache_path: str = get_search_cache_path(image_request, color_spaces, descriptors, normalization, distance, max_results)
		if os.path.exists(cache_path):
			with open(cache_path, 'rb') as f:
				return [(path, None, dist) for path, dist in pickle.load(f)]

	# Check if the color spaces, descriptors and distance are valid
	for color_space in color_spaces:
		assert color_space in COLOR_SPACES_CALLS, f"Color space '{color_space}' not found in {list(COLOR_SPACES_CALLS.keys())}"
	for descriptor in descriptors:
		assert descriptor in DESCRIPTORS_CALLS, f"Descriptor '{descriptor}' not found in {list(DESCRIPTORS_CALLS.keys())}"
	assert normalization in NORMALIZATION_CALLS, f"Normalization '{normalization}' not found in {list(NORMALIZATION_CALLS.keys())}"
	assert distance in DISTANCES_CALLS, f"Distance '{distance}' not found in {list(DISTANCES_CALLS.keys())}"

	# If the image is a path, load it
	if isinstance(image_request, str):
		img: ImageData = load_and_process_image(image_request, color_spaces, descriptors, normalization)
		if not img:
			return []
	elif isinstance(image_request, Image.Image):
		image_request = resize_down(image_request)
		img: ImageData = ImageData(img_to_sliced_rgb(np.array(image_request)), "RGB")
	
		# Apply each color space in sequence
		for color_space in color_spaces:
			cs: dict = COLOR_SPACES_CALLS[color_space]
			cs_args: dict = cs.get("args", {})
			img = cs["function"](img, **cs_args)

		# Apply descriptors in parallel and concatenate results
		descriptor_results = []
		for descriptor in descriptors:
			ds: dict = DESCRIPTORS_CALLS[descriptor]
			ds_args: dict = ds.get("args", {})
			result = ds["function"](img, **ds_args)
			descriptor_results.append(result.data)
		
		# Concatenate all descriptor results
		if descriptor_results:
			img = ImageData(np.concatenate(descriptor_results), "Descriptor")
	
		# Apply normalization
		norm: dict = NORMALIZATION_CALLS[normalization]
		img.data = norm["function"](img.data, **norm.get("args", {}))

	# Apply the color spaces and descriptors to the images, then compute the distance
	images_paths: list[str] = [f"{root}/{file}" for root, _, files in os.walk(DATABASE_FOLDER) 
							for file in files if file.endswith(IMAGE_EXTENSIONS)]
	thread_args: list[tuple] = [(image_path, color_spaces, descriptors, normalization, distance, img.data) 
							for image_path in images_paths]
	
	# Choose parallelization method
	if parallel == "process" and cpu_count() > 4:
		with Pool(cpu_count()) as pool:
			results: list[tuple[str, float]] = pool.starmap(thread_function, thread_args)
	elif parallel == "thread":
		with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
			results: list[tuple[str, float]] = list(executor.map(lambda x: thread_function(*x), thread_args))
	else:
		results: list[tuple[str, float]] = [thread_function(*args) for args in thread_args]
	
	# Filter None results
	results = [result for result in results if result is not None]

	# Sort the images by distance and return the most similar ones
	sorted_images: list[tuple[str, float]] = sorted(results, key=lambda x: x[-1])[:max_results]
	final_results = [(image_path, Image.open(image_path).convert("RGB"), distance) 
			for image_path, distance in sorted_images]

	# Cache the results if the input was a path (only paths and distances)
	if isinstance(image_request, str):
		os.makedirs(os.path.dirname(cache_path), exist_ok=True)
		if not os.path.exists(cache_path):
			# Store only paths and distances, not images
			with open(cache_path, 'wb') as f:
				pickle.dump(sorted_images, f)

	return final_results

def lqdm_call(args: tuple) -> tuple:
	return thread_function(*args)

# Function to compute every single cache
def offline_cache_compute() -> None:
	""" Compute every single cache for the database """
	# Apply the color spaces and descriptor to the images
	images_paths: list[str] = [f"{root}/{file}" for root, _, files in os.walk(DATABASE_FOLDER) for file in files if file.endswith(IMAGE_EXTENSIONS)]

	# Prepare arguments
	thread_args: list[tuple] = []
	for image_path in images_paths:
		for color_space in COLOR_SPACES_CALLS:
			# thread_args.append((image_path, [color_space], [], None, None))
			for descriptor in DESCRIPTORS_CALLS:
				for normalization in NORMALIZATION_CALLS:
					# img, color_spaces, descriptors, normalization, distance
					thread_args.append((image_path, [color_space], [descriptor], normalization, None))
	
	# Compute the cache (using multiprocessing if available)
	if DO_MULTI_PROCESSING:
		with Pool(cpu_count()) as pool:
			debug(f"Using {cpu_count()} processes to compute {len(thread_args)} images caches")
			from tqdm import tqdm
			list(tqdm(pool.imap(lqdm_call, thread_args), total=len(thread_args)))
	else:
		debug(f"Computing {len(thread_args)} images caches")
		for args in thread_args:
			thread_function(*args, verbose=True)
	debug(f"Computed {len(thread_args)} images caches")

