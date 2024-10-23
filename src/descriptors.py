
# Imports
import numpy as np
from scipy.ndimage import convolve, prewitt, sobel

# Histogram on a grayscale
def histogram_single_channel(image: np.ndarray, value_range: tuple[float] = (0,256,1), do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a single channel image.\n
	Args:
		image			(np.ndarray):	Single channel image, example shape: (100, 100)
		value_range		(tuple):		Range of the histogram, default is (0, 256, 1)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	nb_classes: int = int((value_range[1] - value_range[0]) // value_range[2])
	histogram: np.ndarray = np.histogram(image.flatten(), bins=nb_classes, range=value_range[:2])[0]
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

# Blob histogram #TODO: MODIFIER LA VALUE_RANGE POUR LA VRAIE RANGE VENANT DU PREVIOUS COLOR_SPACE
def histogram_blob(image: np.ndarray, blob_size: tuple[int,int] = (4,4), quantifiers: int = 4, value_range: tuple[float,float,float] = (0,256,1), do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a 2D image with blobs.\n
	Warning: This is a very slow function!\n
	Args:
		image			(np.ndarray):	2D image, example shape: (100, 100)
		blob_size		(tuple):		Size of the blob, default is (4, 4)
		quantifiers		(int):			Number of classes for the histogram, default is 4 (ex: 0-25%, 2-50%, ...)
		value_range		(tuple):		Range of the small histogram, default is (0, 256, 1)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	# If not grayscale, call the function for each channel
	if len(image.shape) > 2:
		histograms: list[np.ndarray] = [histogram_blob(image[i], blob_size, quantifiers, value_range, do_normalize) for i in range(image.shape[0])]
		return np.concatenate(histograms)

	# Assertions
	assert blob_size[0] < image.shape[0] and blob_size[1] < image.shape[1], f"Blob size must be smaller than the image, got {blob_size} and {image.shape}"

	# Compute the histogram using the blob
	nb_classes: int = int((value_range[1] - value_range[0]) // value_range[2])	# (max-min) // step
	value_range_2: tuple[float] = (value_range[0], value_range[1])
	histogram: np.ndarray = np.zeros((nb_classes, quantifiers))
	for i in range(0, image.shape[0] - blob_size[0]):
		for j in range(0, image.shape[1] - blob_size[1]):
			blob: np.ndarray = image[i:i+blob_size[0], j:j+blob_size[1]]

			# Compute the histogram of the blob
			#blob_h: np.ndarray = histogram_single_channel(blob, value_range, do_normalize=True)
			blob_h: np.ndarray = np.histogram(blob.flatten(), bins=nb_classes, range=value_range_2)[0]
			blob_h = blob_h / np.sum(blob_h)

			# Add the histogram of the blob to the global histogram
			for k in range(len(blob_h)):
				column: int = int(blob_h[k] * quantifiers)
				if column == quantifiers:	# If the value is 1.0, we need to decrement the column
					column -= 1
				histogram[k, column] += 1
	
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


## Formes
FILTERS: dict[str, np.ndarray] = {
	"prewitt": np.array([
		[
			[-1, 0, 1],
			[-1, 0, 1],
			[-1, 0, 1],
		],[
			[1, 1, 1],
			[0, 0, 0],
			[-1, -1, -1],
		],
	]),
	"sobel": np.array([
		[
			[-1, 0, 1],
			[-2, 0, 2],
			[-1, 0, 1],
		],[
			[-1, -2, -1],
			[0, 0, 0],
			[1, 2, 1],
		],
	]),
	"scharr": np.array([
		[
			[-3, 0, 3],
			[-10, 0, 10],
			[-3, 0, 3],
		],[
			[-3, -10, -3],
			[0, 0, 0],
			[3, 10, 3],
		],
	]),
}

# Calculate dx and dy
def compute_dx_dy(image: np.ndarray, filter_name: str, crop: bool = True) -> tuple[np.ndarray, np.ndarray]:
	""" Compute the dx and dy images using a filter.\n
	Args:
		image		(np.ndarray):	Image
		filter_name	(str):			Name of the filter
		crop		(bool):			Crop the resulting image of convolve function, default is True
	Returns:
		tuple[np.ndarray, np.ndarray]: Tuple of dx and dy images
	"""
	# Assertions
	assert filter_name in FILTERS, f"Filter name must be in {list(FILTERS.keys())}, got '{filter_name}'"
	
	# Get the filter
	f: np.ndarray = FILTERS[filter_name]
	f_shape: int = np.prod(f.shape)

	# Apply the filter
	if crop:
		np.convolve()
		dx: np.ndarray = convolve(image, f[0])[1:-1, 1:-1] / f_shape
		dy: np.ndarray = convolve(image, f[1])[1:-1, 1:-1] / f_shape
	else:
		dx: np.ndarray = convolve(image, f[0]) / f_shape
		dy: np.ndarray = convolve(image, f[1]) / f_shape
	return dx, dy


# Name every function
from typing import Callable
DESCRIPTORS_CALLS: dict[str, Callable] = {
	# Histograms
	"Histogram":			{"function":histogram_multi_channels, "args":{}},
	"Histogram (HSV/HSL)":	{"function":histogram_hue_per_saturation, "args":{}},
	"Histogram Blob":		{"function":histogram_blob, "args":{}},

	# Statistics (mean, median, std, Q1, Q3)
	"Mean":					{"function":mean, "args":{}},
	"Median":				{"function":median, "args":{}},
	"Std (écart-type)":		{"function":std, "args":{}},
	"Q1":					{"function":Q1, "args":{}},
	"Q3":					{"function":Q3, "args":{}},
}

