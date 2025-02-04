
# Imports
import numpy as np
from src.image import ImageData

# Utility function to generate a random image
def random_image(size: int, seed: int|None = None, maxi: int = 256) -> ImageData:
	""" Generate a random image of size `size`x`size`\n
	Args:
		size	(int):	Size of the image
		seed	(int):	Seed for random number generator (default: None)
		maxi	(int):	Maximum value for the randint function
	Returns:
		(ImageData): Random image (2D array)
	"""
	if seed is not None:
		np.random.seed(seed)
	return ImageData(np.random.randint(0, maxi, (3, size, size)), "RGB")

# Convert an image to a sliced RGB image
def img_to_sliced_rgb(image: np.ndarray) -> np.ndarray:
	""" Slice the image in 3 images (each color)\n
	Args:
		image	(np.ndarray):	RGB image (3D array)
	Returns:
		(np.ndarray): Sliced RGB image (3D array)
	"""
	assert isinstance(image, np.ndarray), "Image must be a numpy array"
	return np.moveaxis(image, -1, 0)

# Convert a sliced RGB image to an image
def sliced_rgb_to_img(sliced_image: np.ndarray) -> np.ndarray:
	""" Merge the 3 images in one\n
	Args:
		sliced_image	(np.ndarray):	Sliced RGB image (3D array)
	Returns:
		(np.ndarray): RGB image (3D array)
	"""
	return np.moveaxis(sliced_image, 0, -1)

# Function to return input
def same_color_space(image):
	return image

