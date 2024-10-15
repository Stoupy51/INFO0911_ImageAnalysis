
# Imports
import numpy as np

# Histogram on a grayscale
def histogram_single_channel(image: np.ndarray, range: tuple[float,float] = (0,256), do_normalize: bool = False) -> np.ndarray:
	""" Compute the histogram vector of a single channel image.\n
	Args:
		image			(np.ndarray):	Single channel image, example shape: (100, 100)
		range			(tuple):		Range of the histogram, default is (0, 256)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is False
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	histogram: np.ndarray = np.histogram(image, bins=(range[1] - range[0]), range=range)[0]
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return histogram

# Histogram on multiple channels
def histogram_multi_channels(image: np.ndarray, range: tuple[float,float] = (0,256), do_normalize: bool = False) -> np.ndarray:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		image			(np.ndarray):	Multi channel image, example shape: (3, 100, 100)
		range			(tuple):		Range of the histogram, default is (0, 256)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is False
	Returns:
		np.ndarray: 1 dimension array, example shape: (256*3,)
	"""
	histograms: list[np.ndarray] = [histogram_single_channel(channel, range, do_normalize) for channel in image]
	return np.concatenate(histograms)


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
	"Histogram single channel":	histogram_single_channel,
	"Histogram multi channels":	histogram_multi_channels,

	# Statistics (mean, median, std, Q1, Q3)
	"Mean":					mean,
	"Median":				median,
	"Std (écart-type)":		std,
	"Q1":					Q1,
	"Q3":					Q3
}

