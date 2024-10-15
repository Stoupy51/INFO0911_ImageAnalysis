
# Imports
from src.color_space.all import *
from src.descriptors import *
from src.distances import *
from src.print import *
from config import *

COLOR_SPACES_CALLS: dict[str, Callable] = {...}

# Search engine
def search(image_request: np.ndarray, color_space: str, descriptor: str, distance: str, max_results: int = 5) -> list[np.ndarray]:
	""" Search for similar images in the database\n
	Args:
		image_request	(np.ndarray):	Image to search for, example shape: (100, 100, 3)
		color_space		(str):			Color space to use for the descriptor
		descriptor		(str):			Descriptor to use for the search
		distance		(str):			Distance to use for the search
		max_results		(int):			Maximum number of results to return, default is 5
	Returns:
		list[np.ndarray]: List of the most similar images found, example shape: [(100, 100, 3), ...]
	"""
	# Check if the color space, descriptor and distance are valid
	assert color_space in COLOR_SPACES_CALLS, f"Color space '{color_space}' not found in {list(COLOR_SPACES_CALLS.keys())}"
	assert descriptor in DESCRIPTORS_CALLS, f"Descriptor '{descriptor}' not found in {list(DESCRIPTORS_CALLS.keys())}"
	assert distance in DISTANCES_CALLS, f"Distance '{distance}' not found in {list(DISTANCES_CALLS.keys())}"

	# TODO
	pass


