
# Imports
import numpy as np

# Histogram on a grayscale
def histogram_single_channel(image: np.ndarray, nb_classes: int = 256, range: tuple[float,float] = (0,256), do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a single channel image.\n
	Args:
		image			(np.ndarray):	Single channel image, example shape: (100, 100)
		nb_classes		(int):			Number of classes for the histogram, default is 256
		range			(tuple):		Range of the histogram, default is (0, 256)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	histogram: np.ndarray = np.histogram(image.flatten(), bins=nb_classes, range=range)[0]
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return histogram

# Histogram on multiple channels
def histogram_multi_channels(image: np.ndarray, nb_classes: list[int] = 3*[256], ranges: list[tuple[float,float]] = 3*[(0,256)], do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		image			(np.ndarray):	Multi channel image, example shape: (3, 100, 100)
		nb_classes		(list[int]):	Number of classes for each channel, default is [256, 256, 256]
		ranges			(list[tuple]):	Range of the histogram for each channel, default is [(0, 256), (0, 256), (0, 256)]
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256*3,)
	"""
	# Grayscale input
	if len(image.shape) == 2:
		return histogram_single_channel(image, nb_classes[0], ranges[0], do_normalize)
	histograms: list[np.ndarray] = [histogram_single_channel(image[i], nb_classes[i], ranges[i], do_normalize) for i in range(image.shape[0])]
	return np.concatenate(histograms)

# Histogram on HSV or HSL
def histogram_hue_per_saturation(image: np.ndarray, do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		image			(np.ndarray):	HSV or HSL image, example shape: (3, 100, 100)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (360)
	"""
	# Assertions
	assert image.shape[0] == 3, "Image must be in HSV or HSL format"
	assert len(image.shape) == 3, "Image must be 3D"
	
	# Get the hue and saturation channels
	hue: np.ndarray = image[0]
	saturation: np.ndarray = image[1]
	
	# Compute the histogram
	histogram: np.ndarray = np.zeros((360,))
	for i in range(image.shape[1]):
		for j in range(image.shape[2]):
			histogram[int(hue[i,j])] += saturation[i,j]
	
	# Normalize the histogram
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return histogram

	

# Statistics
def mean(image: np.ndarray) -> float:
	""" Compute the mean of the image.\n
	Args:
		image	(np.ndarray):	Image
	Returns:
		float: Mean value
	"""
	return np.mean(image)

def median(image: np.ndarray) -> float:
	""" Compute the median of the image.\n
	Args:
		image	(np.ndarray):	Image
	Returns:
		float: Median value
	"""
	return np.median(image)

def std(image: np.ndarray) -> float:
	""" Compute the standard deviation of the image.\n
	Args:
		image	(np.ndarray):	Image
	Returns:
		float: Standard deviation value
	"""
	return np.std(image)

def Q1(image: np.ndarray) -> float:
	""" Compute the first quartile of the image.\n
	Args:
		image	(np.ndarray):	Image
	Returns:
		float: First quartile value
	"""
	return np.percentile(image, 25)

def Q3(image: np.ndarray) -> float:
	""" Compute the third quartile of the image.\n
	Args:
		image	(np.ndarray):	Image
	Returns:
		float: Third quartile value
	"""
	return np.percentile(image, 75)


# Name every function
from typing import Callable
DESCRIPTORS_CALLS: dict[str, Callable] = {
	# Histograms
	"Histogram":			{"function":histogram_multi_channels, "args":{}},
	"Histogram HSV/HSL":	{"function":histogram_hue_per_saturation, "args":{}},	

	# Statistics (mean, median, std, Q1, Q3)
	"Mean":					{"function":mean, "args":{}},
	"Median":				{"function":median, "args":{}},
	"Std (écart-type)":		{"function":std, "args":{}},
	"Q1":					{"function":Q1, "args":{}},
	"Q3":					{"function":Q3, "args":{}},
}

