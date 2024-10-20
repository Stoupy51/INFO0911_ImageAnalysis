
# Imports
import numpy as np

# Histogram on a grayscale
def histogram_single_channel(image: np.ndarray, range: tuple[float] = (0,256,1), do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a single channel image.\n
	Args:
		image			(np.ndarray):	Single channel image, example shape: (100, 100)
		range			(tuple):		Range of the histogram, default is (0, 256, 1)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	nb_classes: int = int((range[1] - range[0]) // range[2])
	histogram: np.ndarray = np.histogram(image.flatten(), bins=nb_classes, range=range[:2])[0]
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return histogram

# Histogram on multiple channels
def histogram_multi_channels(image: np.ndarray, ranges: list[tuple[float]] = 3*[(0,256,1)], do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		image			(np.ndarray):	Multi channel image, example shape: (3, 100, 100)
		ranges			(list[tuple]):	Range of the histogram for each channel, default is [(0,256,1), (0,256,1), (0,256,1)]
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256*3,)
	"""
	# Grayscale input
	if len(image.shape) == 2:
		return histogram_single_channel(image, ranges[0], do_normalize)
	histograms: list[np.ndarray] = [histogram_single_channel(image[i], ranges[i], do_normalize) for i in range(image.shape[0])]
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

# TODO: range(start, stop) => range(start,stop,step) so I can remove the nb_classes argument

# Blob histogram
def histogram_blob_2D(image: np.ndarray, blob_size: tuple[int,int] = (4,4), quantifiers: int = 4, range: tuple[float,float,float] = (0,256,1), do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a 2D image with blobs.\n
	Args:
		image			(np.ndarray):	2D image, example shape: (100, 100)
		blob_size		(tuple):		Size of the blob, default is (4, 4)
		quantifiers		(int):			Number of classes for the histogram, default is 4 (ex: 0-25%, 2-50%, ...)
		range			(tuple):		Range of the small histogram, default is (0, 256, 1)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	# Assertions
	assert len(image.shape) == 2, f"Image must be 2D, got {image.shape}"
	assert blob_size[0] < image.shape[0] and blob_size[1] < image.shape[1], f"Blob size must be smaller than the image, got {blob_size} and {image.shape}"

	# Compute the histogram using the blob
	nb_classes: int = int((range[1] - range[0]) // range[2])	# (max-min) // step
	histogram: np.ndarray = np.zeros((nb_classes, quantifiers))
	for i in range(0, image.shape[0] - blob_size[0]):
		for j in range(0, image.shape[1] - blob_size[1]):
			blob: np.ndarray = image[i:i+blob_size[0], j:j+blob_size[1]]

			# Compute the histogram of the blob
			histogram_blob: np.ndarray = histogram_single_channel(blob, range, do_normalize=True)

			# Add the histogram of the blob to the global histogram
			for v in histogram_blob:
				column: int = int(v*quantifiers)
				histogram[v, column] += 1
	
	# Normalize the histogram and return it
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return histogram.flatten()


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
	"Histogram Blob":		{"function":histogram_blob_2D, "args":{}},

	# Statistics (mean, median, std, Q1, Q3)
	"Mean":					{"function":mean, "args":{}},
	"Median":				{"function":median, "args":{}},
	"Std (écart-type)":		{"function":std, "args":{}},
	"Q1":					{"function":Q1, "args":{}},
	"Q3":					{"function":Q3, "args":{}},
}

