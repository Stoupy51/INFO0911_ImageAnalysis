
# Imports
import numpy as np


# Image indexation on a grayscale
def indexed_single_channel(image: np.ndarray, levels: int, range: tuple[float, float] = (0, 256)) -> np.ndarray:
	""" Index an image to a specific number of levels\n
	Args:
		image	(np.ndarray):	Single image channel to quantify, example shape: (100, 100)
		levels	(int):			Number of levels to quantize the image
		range	(tuple):		Range of the image, default is (0, 256)
	Returns:
		np.ndarray: Indexed image, example shape: (100, 100) with values between 0 and levels
	"""
	# Basic formula: floor(image / (256 / levels))
	# Ex: 256 / 8 = 32 and 127 / 32 = 3.96875 -> 3
	mini, maxi = range
	step = (maxi - mini) / levels	# Step size, ex: 256 / 8 = 32
	return ((image - mini) // step).astype(int)


# Image indexation on multiple channels
def indexed_multi_channels(image: np.ndarray, levels: list[int], ranges: list[tuple] = 3*[(0, 256)]) -> np.ndarray:
	""" Index a multi channel image to a specific number of levels\n
	Args:
		image	(np.ndarray):	Multi channel image to quantify, example shape: (3, 100, 100)
		levels	(list[int]):	Number of levels to quantize each channel
		ranges	(list[tuple]):	Range of each channel, default is [(0, 256), (0, 256), (0, 256)]
	Returns:
		np.ndarray: Indexed image, example shape: (3, 100, 100) with values between 0 and levels
		list[np.ndarray]: List of every indexed channel from the base image
	"""
	# Assertions
	assert image.shape[0] == len(levels), "Number of levels must be equal to the number of channels"
	assert image.shape[0] == len(ranges), "Number of ranges must be equal to the number of channels"

	# Index each channel
	indexed_channels: list[np.ndarray] = [
		indexed_single_channel(image[i], levels[i], ranges[i]) for i in range(image.shape[0])
	]

	# Combine the indexed channels (ex: 3D -> 2D for RGB)
	# formula for RGB: q = qR + qG*levels[0] + qB*levels[0]*levels[1]
	indexed: np.ndarray = np.zeros(image.shape[1:], dtype=int)
	multiplier: int = 1
	for i in range(image.shape[0]):
		indexed += indexed_channels[i] * multiplier
		multiplier *= levels[i]		# Example: i=0 -> 1*2=2,	i=1 -> 2*3=6	(if levels=[2, 3, 4])
	
	# Return the indexed image
	return indexed, indexed_channels

