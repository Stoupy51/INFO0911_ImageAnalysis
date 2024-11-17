
# Imports
import numpy as np
from src.image import ImageData
from scipy.ndimage import convolve, prewitt, sobel

# Histogram on a grayscale
def histogram_single_channel(img: ImageData, do_normalize: bool = True) -> ImageData:
	""" Compute the histogram vector of a single channel image.\n
	Args:
		img				(ImageData):	Single channel image, example shape: (100, 100)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		ImageData: 1 dimension array, example shape: (256,)
	"""
	value_range: tuple[float,float,float] = img.range
	nb_classes: int = int((value_range[1] - value_range[0]) // value_range[2])
	histogram: np.ndarray = np.histogram(img.data.flatten(), bins=nb_classes, range=value_range[:2])[0]
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return ImageData(histogram, img.color_space, img.channel)

# Histogram on multiple channels
def histogram_multi_channels(img: ImageData, do_normalize: bool = True) -> ImageData:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		img				(ImageData):	Multi channel image, example shape: (3, 100, 100)
		ranges			(list[tuple]):	Range of the histogram for each channel, default is [(0,256,1), (0,256,1), (0,256,1)]
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256*3,)
	"""
	# Grayscale input
	if len(img.shape) == 2:
		return histogram_single_channel(img, do_normalize)
	histograms: list[np.ndarray] = [histogram_single_channel(img[i], do_normalize).data for i in range(img.shape[0])]
	return ImageData(np.concatenate(histograms), img.color_space, img.channel)

# Histogram on HSV or HSL
def histogram_hue_per_saturation(img: ImageData, do_normalize: bool = True) -> ImageData:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		img				(ImageData):	HSV or HSL image, example shape: (3, 100, 100)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (360)
	"""
	# Assertions
	assert img.color_space in ["HSV", "HSL"], "Image must be in HSV or HSL format"
	assert len(img.shape) == 3, "Image must be 3D"
	
	# Get the hue and saturation channels
	hue: ImageData = img[0]
	saturation: ImageData = img[1]
	
	# Compute the histogram
	histogram: np.ndarray = np.zeros((360,))
	for i in range(img.shape[1]):
		for j in range(img.shape[2]):
			histogram[int(hue.data[i,j])] += saturation.data[i,j]
	
	# Normalize the histogram
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return ImageData(histogram, img.color_space, img.channel)

# Blob histogram
def histogram_blob(img: ImageData, blob_size: tuple[int,int] = (4,4), quantifiers: int = 4, do_normalize: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a 2D image with blobs.\n
	Warning: This is a very slow function!\n
	Args:
		img				(ImageData):	2D image, example shape: (100, 100)
		blob_size		(tuple):		Size of the blob, default is (4, 4)
		quantifiers		(int):			Number of classes for the histogram, default is 4 (ex: 0-25%, 2-50%, ...)
		do_normalize	(bool):			Normalize the histogram vector (sum to 1), default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	# If not grayscale, call the function for each channel
	if len(img.shape) > 2:
		histograms: list[np.ndarray] = [histogram_blob(img[i], blob_size, quantifiers, do_normalize).data for i in range(img.shape[0])]
		return ImageData(np.concatenate(histograms), img.color_space, img.channel)

	# Assertions
	assert blob_size[0] < img.shape[0] and blob_size[1] < img.shape[1], f"Blob size must be smaller than the image, got {blob_size} and {img.shape}"

	# Get value_range
	value_range: tuple[int, ...] = img.range

	# Compute the histogram using the blob
	nb_classes: int = int((value_range[1] - value_range[0]) // value_range[2])	# (max-min) // step
	value_range_2: tuple[float, float] = (value_range[0], value_range[1])
	histogram: np.ndarray = np.zeros((nb_classes, quantifiers))
	for i in range(0, img.shape[0] - blob_size[0]):
		for j in range(0, img.shape[1] - blob_size[1]):
			blob: np.ndarray = img.data[i:i+blob_size[0], j:j+blob_size[1]]

			# Compute the histogram of the blob
			#blob_h: np.ndarray = histogram_single_channel(blob, value_range, do_normalize=True)
			blob_h: np.ndarray = np.histogram(blob.flatten(), bins=nb_classes, range=value_range_2)[0]
			sum_blob_h: float = np.sum(blob_h)
			if sum_blob_h == 0:
				continue
			blob_h = blob_h / sum_blob_h

			# Add the histogram of the blob to the global histogram
			for k in range(len(blob_h)):
				if np.isnan(blob_h[k]):
					continue
				column: int = int(blob_h[k] * quantifiers)
				if column == quantifiers:	# If the value is 1.0, we need to decrement the column
					column -= 1
				histogram[k, column] += 1
	
	# Normalize the histogram and return it
	if do_normalize:
		histogram = histogram / np.sum(histogram)
	return ImageData(histogram.flatten(), img.color_space, img.channel)


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
def compute_dx_dy(image: np.ndarray, filter_name: str = "sobel", crop: bool = True) -> tuple[np.ndarray, np.ndarray]:
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

# Horizontal gradient
def horizontal_gradient(img: ImageData, filter_name: str = "sobel") -> ImageData:
	dx, _ = compute_dx_dy(img, filter_name)
	return ImageData(dx, img.color_space, img.channel)

# Vertical gradient
def vertical_gradient(img: ImageData, filter_name: str = "sobel") -> ImageData:
	_, dy = compute_dx_dy(img, filter_name)
	return ImageData(dy, img.color_space, img.channel)

# Gradient norm
def gradient_norm(img: ImageData, filter_name: str = "sobel") -> ImageData:
	dx, dy = compute_dx_dy(img, filter_name)
	return ImageData(np.sqrt(dx**2 + dy**2), img.color_space, img.channel)

# Gradient orientation
def gradient_orientation(img: ImageData, filter_name: str = "sobel") -> ImageData:
	dx, dy = compute_dx_dy(img, filter_name)
	orientation: np.ndarray = (np.arctan2(dy, dx) * 180 / np.pi) % 360
	orientation = np.where(dx == 0, 0, orientation)
	orientation = np.where(dy == 0, 0, orientation)
	return ImageData(orientation, img.color_space, img.channel)


## Textures
def statistics(img: ImageData) -> ImageData:
	""" Compute the statistics of the image.\n
	Args:
		img		(ImageData):	Image
	Returns:
		ImageData: Array of statistics (mean, median, std, min, max, Q1, Q3)
	"""
	return ImageData(np.array([
		np.mean(img.data),
		np.median(img.data),
		np.std(img.data),
		np.min(img.data),
		np.max(img.data),
		np.percentile(img.data, 25),
		np.percentile(img.data, 75),
	]), "Statistics")



# Name every function
from typing import Callable
DESCRIPTORS_CALLS: dict[str, Callable] = {
	# Histograms
	"Histogram":			{"function":histogram_multi_channels, "args":{}},
	"Histogram (HSV/HSL)":	{"function":histogram_hue_per_saturation, "args":{}},
	"Histogram Blob":		{"function":histogram_blob, "args":{}},

	# Formes
	"Horizontal Gradient":	{"function":horizontal_gradient, "args":{}},
	"Vertical Gradient":	{"function":vertical_gradient, "args":{}},
	"Gradient Norm":		{"function":gradient_norm, "args":{}},
	"Gradient Orientation":	{"function":gradient_orientation, "args":{}},

	# Textures
	"Statistics":			{"function":statistics, "args":{}},
}

