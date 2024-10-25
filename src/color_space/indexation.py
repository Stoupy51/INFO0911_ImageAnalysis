
# Imports
import numpy as np
from src.image import ImageData

# Image indexation on a grayscale
def indexed_single_channel(img: ImageData, levels: int = 8) -> ImageData:
	""" Index an image to a specific number of levels\n
	Args:
		img		(np.ndarray):	Single image channel to quantify, example shape: (100, 100)
		levels	(int):			Number of levels to quantize the image, default is 8
	Returns:
		ImageData: Indexed image, example shape: (100, 100) with values between 0 and levels
	"""
	# Basic formula: floor(image / (256 / levels))
	# Ex: 256 / 8 = 32 and 127 / 32 = 3.96875 -> 3
	mini, maxi = img.range[:2]	# Minimum and maximum values of the image
	step = (maxi - mini) / levels	# Step size, ex: 256 / 8 = 32
	new_image: np.ndarray = ((img.data - mini) // step).astype(int)
	return ImageData(new_image, f"Indexation ({levels})")


# Image indexation on multiple channels
def indexed_multi_channels(img: ImageData, levels: list[int] = [8, 8, 8]) -> ImageData:
	""" Index a multi channel image to a specific number of levels\n
	Args:
		img		(ImageData):	Multi channel image to quantify, example shape: (3, 100, 100)
		levels	(list[int]):	Number of levels to quantize each channel, default is [8, 8, 8]
	Returns:
		ImageData: Indexed image, example shape: (3, 100, 100) with values between 0 and levels
	"""
	# Grayscale input
	if len(img.data.shape) == 2:
		return indexed_single_channel(img, levels[0])

	# Assertions
	assert img.data.shape[0] == len(levels), "Number of levels must be equal to the number of channels"

	# Index each channel
	indexed_channels: list[ImageData] = [
		indexed_single_channel(img[i], levels[i]) for i in range(img.data.shape[0])
	]

	# Combine the indexed channels (ex: 3D -> 2D for RGB)
	# formula for RGB: q = qR + qG*levels[0] + qB*levels[0]*levels[1]
	indexed: np.ndarray = np.zeros(img.data.shape[1:], dtype=int)
	multiplier: int = 1
	for i in range(img.data.data.shape[0]):
		indexed += indexed_channels[i] * multiplier
		multiplier *= levels[i]		# Example: i=0 -> 1*2=2,	i=1 -> 2*3=6	(if levels=[2, 3, 4])
	
	# Return the indexed image
	return ImageData(indexed, "Indexation (" + ",".join(levels) + ')')

