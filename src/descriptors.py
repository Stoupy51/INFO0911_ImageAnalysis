
# Imports
import numpy as np
from src.image import ImageData
from scipy.ndimage import convolve
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'					# Suppress TF logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)		# Suppress TF warnings

# Histogram on a grayscale
def histogram_single_channel(img: ImageData) -> ImageData:
	""" Compute the histogram vector of a single channel image.\n
	Args:
		img				(ImageData):	Single channel image, example shape: (100, 100)
	Returns:
		ImageData: 1 dimension array, example shape: (256,)
	"""
	value_range: tuple[float,float,float] = img.range
	nb_classes: int = int((value_range[1] - value_range[0]) // value_range[2])
	histogram: np.ndarray = np.histogram(img.data.flatten(), bins=nb_classes, range=value_range[:2])[0]
	return ImageData(histogram, img.color_space, img.channel)

# Histogram on multiple channels
def histogram_multi_channels(img: ImageData) -> ImageData:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		img				(ImageData):	Multi channel image, example shape: (3, 100, 100)
		ranges			(list[tuple]):	Range of the histogram for each channel, default is [(0,256,1), (0,256,1), (0,256,1)]
	Returns:
		np.ndarray: 1 dimension array, example shape: (256*3,)
	"""
	# Grayscale input
	if len(img.shape) == 2:
		return histogram_single_channel(img)
	histograms: list[np.ndarray] = [histogram_single_channel(img[i]).data for i in range(img.shape[0])]
	return ImageData(np.concatenate(histograms), img.color_space, img.channel)

# Histogram on HSV or HSL
def histogram_hue_per_saturation(img: ImageData) -> ImageData:
	""" Compute the histogram vector of a multi channel image.\n
	Args:
		img				(ImageData):	HSV or HSL image, example shape: (3, 100, 100)
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
	
	return ImageData(histogram, img.color_space, img.channel)

# Blob histogram
def histogram_blob(img: ImageData, blob_size: tuple[int,int] = (4,4), quantifiers: int = 4, optimized: bool = True) -> np.ndarray:
	""" Compute the histogram vector of a 2D image with blobs.\n
	Warning: This is a very slow function if not optimized!\n
	Args:
		img				(ImageData):	2D image, example shape: (100, 100)
		blob_size		(tuple):		Size of the blob, default is (4, 4)
		quantifiers		(int):			Number of classes for the histogram, default is 4 (ex: 0-25%, 2-50%, ...)
		optimized		(bool):			Use optimized version with numpy operations, default is True
	Returns:
		np.ndarray: 1 dimension array, example shape: (256,)
	"""
	# If not grayscale, call the function for each channel
	if len(img.shape) > 2:
		histograms: list[np.ndarray] = [histogram_blob(img[i], blob_size, quantifiers, optimized).data for i in range(img.shape[0])]
		return ImageData(np.concatenate(histograms), img.color_space, img.channel)

	# Assertions
	assert blob_size[0] < img.shape[0] and blob_size[1] < img.shape[1], f"Blob size must be smaller than the image, got {blob_size} and {img.shape}"

	# Get value_range
	value_range: tuple[int, ...] = img.range

	# Compute the histogram using the blob
	nb_classes: int = int((value_range[1] - value_range[0]) // value_range[2])	# (max-min) // step
	value_range_2: tuple[float, float] = (value_range[0], value_range[1])
	histogram: np.ndarray = np.zeros((nb_classes, quantifiers))

	if optimized:
		# Create all possible blobs using stride tricks
		shape: tuple[int, int, int, int] = ((img.shape[0] - blob_size[0]), (img.shape[1] - blob_size[1]), blob_size[0], blob_size[1])
		strides: tuple[int, int, int, int] = (img.data.strides[0], img.data.strides[1], img.data.strides[0], img.data.strides[1])
		blocks: np.ndarray = np.lib.stride_tricks.as_strided(img.data, shape=shape, strides=strides)
		
		# Reshape blocks to 2D array where each row is a flattened blob
		blocks = blocks.reshape(-1, blob_size[0] * blob_size[1])
		
		# Compute histograms for all blocks at once
		blob_histograms: np.ndarray = np.apply_along_axis(
			lambda x: np.histogram(x, bins=nb_classes, range=value_range_2)[0], 
			1, blocks
		)
		
		# Normalize histograms
		sums: np.ndarray = np.sum(blob_histograms, axis=1)
		valid_mask: np.ndarray = sums != 0
		blob_histograms = blob_histograms.astype(np.float64)  # Convert to float64 before division
		blob_histograms[valid_mask] = blob_histograms[valid_mask] / sums[valid_mask, np.newaxis]
		
		# Calculate quantifier columns for each histogram
		columns: np.ndarray = np.clip((blob_histograms * quantifiers).astype(int), 0, quantifiers-1)
		
		# Add to histogram using numpy operations
		for k in range(nb_classes):
			for q in range(quantifiers):
				histogram[k,q] = np.sum((columns[:,k] == q) & valid_mask.flatten())
	else:
		# Use the original loop-based implementation
		for i in range(0, img.shape[0] - blob_size[0]):
			for j in range(0, img.shape[1] - blob_size[1]):
				blob: np.ndarray = img.data[i:i+blob_size[0], j:j+blob_size[1]]

				# Compute the histogram of the blob
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
	
	# Return the histogram
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
		image		(np.ndarray):	Image (2D or 3D)
		filter_name	(str):			Name of the filter
		crop		(bool):			Crop the resulting image of convolve function, default is True
	Returns:
		tuple[np.ndarray, np.ndarray]: Tuple of dx and dy images
	"""
	# Assertions
	assert filter_name in FILTERS, f"Filter name must be in {list(FILTERS.keys())}, got '{filter_name}'"
	
	# Get the filter
	f: np.ndarray = FILTERS[filter_name]
	f_shape: int = np.prod(f[0].shape)  # Use shape of single filter, not whole array

	# Apply the filter
	if crop:
		dx: np.ndarray = convolve(image, f[0])[1:-1, 1:-1] / f_shape
		dy: np.ndarray = convolve(image, f[1])[1:-1, 1:-1] / f_shape
	else:
		dx: np.ndarray = convolve(image, f[0]) / f_shape
		dy: np.ndarray = convolve(image, f[1]) / f_shape
	return dx, dy

# Gradient magnitude (norm)
def gradient_magnitude(img: ImageData, filter_name: str = "sobel") -> ImageData:
	dx, dy = compute_dx_dy(img.data, filter_name)
	return ImageData(np.sqrt(dx**2 + dy**2), img.color_space, img.channel)

# Gradient orientation
def gradient_orientation(img: ImageData, filter_name: str = "sobel") -> ImageData:
	dx, dy = compute_dx_dy(img.data, filter_name)
	orientation: np.ndarray = (np.arctan2(dy, dx) * 180 / np.pi) % 360
	orientation = np.where(dx == 0, 0, orientation)
	orientation = np.where(dy == 0, 0, orientation)
	return ImageData(orientation, img.color_space, img.channel)

# Weighted gradient orientation by magnitude
def weighted_gradient_histogram(img: ImageData, filter_name: str = "sobel") -> ImageData:
	if len(img.shape) > 2:
		images: list[np.ndarray] = [weighted_gradient_histogram(img[i], filter_name).data for i in range(img.shape[0])]
		return ImageData(np.concatenate(images), img.color_space, img.channel)
	magnitude: ImageData = gradient_magnitude(img, filter_name)
	orientation: ImageData = gradient_orientation(img, filter_name)
	return ImageData(magnitude.data * orientation.data, img.color_space, img.channel)

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

# Local Binary Pattern
def local_binary_pattern(img: ImageData, filter_size: int = 3, optimized: bool = True, return_histogram: bool = True) -> ImageData:
	""" Compute the Local Binary Pattern of an image.\n
	Args:
		img					(ImageData):	Image to process
		filter_size			(int):			Size of the filter (must be odd), default is 3
		optimized			(bool):			Whether to use optimized implementation, default is True
		return_histogram	(bool):			Whether to return the histogram of the LBP image, default is True
	Returns:
		ImageData: LBP image or histogram
	"""
	# If not grayscale, recursively call the function for each channel
	if len(img.shape) > 2:
		images: list[np.ndarray] = [local_binary_pattern(img[i], filter_size, optimized).data for i in range(img.shape[0])]
		return ImageData(np.concatenate(images), "LBP")

	# Get image data and dimensions
	image: np.ndarray = img.data
	height, width = image.shape
	
	# Output dimensions after filter
	out_height: int = height - filter_size + 1
	out_width: int = width - filter_size + 1
	
	if optimized:
		# Create views into the image for each pixel's neighborhood
		windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(image, (filter_size, filter_size))
		windows: np.ndarray = windows.reshape(-1, filter_size * filter_size)
		
		# Get center values
		centers: np.ndarray = windows[:, filter_size * filter_size // 2]
		
		# Compare with centers and convert to binary, then remove center values
		binary: np.ndarray = (windows >= centers[:, np.newaxis]).astype(np.uint8)
		binary = np.delete(binary, filter_size * filter_size // 2, axis=1)
		
		# Convert binary patterns to decimal
		powers: np.ndarray = 2**np.arange(filter_size * filter_size - 1)
		dot_product: np.ndarray = binary.dot(powers)
		output: np.ndarray = dot_product.reshape(out_height, out_width)
	else:
		output: np.ndarray = np.zeros((out_height, out_width), dtype=np.uint8)
		
		# Apply LBP filter
		FILTER_SIZE_SQUARED: int = filter_size**2
		for i in range(out_height):
			for j in range(out_width):
				# Get patch and center value
				patch: np.ndarray = image[i:i+filter_size, j:j+filter_size].flatten()
				center: int = patch[FILTER_SIZE_SQUARED//2]
				
				# Compare with center and convert to binary, then remove center value
				binary: np.ndarray = (patch >= center).astype(np.uint8)
				binary = np.delete(binary, FILTER_SIZE_SQUARED//2)
				
				# Convert binary pattern to decimal
				decimal: int = int(binary.dot(2**np.arange(len(binary))))
				output[i,j] = decimal
				
	# Convert to histogram if requested
	if return_histogram:
		nb_possible_patterns: int = 2**(filter_size * filter_size - 1)
		histogram: np.ndarray = np.histogram(output.flatten(), bins=nb_possible_patterns, range=(0, nb_possible_patterns))[0]
		return ImageData(histogram, "LBP Histogram")
	
	# Return the LBP image if histogram not requested
	return ImageData(output.astype(np.uint8), "LBP")

def _create_cooc_matrix(image: np.ndarray, d: int, theta: float, n_gray_levels: int = 256) -> np.ndarray:
	"""Create co-occurrence matrix for given distance and angle.
	Args:
		image			(np.ndarray):	Input image
		d				(int):			Distance to consider
		theta			(float):		Angle in degrees
		n_gray_levels	(int):			Number of gray levels (default: 256)
	Returns:
		np.ndarray: Co-occurrence matrix
	"""
	height, width = image.shape
	
	# Convert angle to radians and calculate offsets
	theta_rad: float = np.radians(theta)
	dx: int = int(d * np.cos(theta_rad))
	dy: int = int(d * np.sin(theta_rad))
	
	# Initialize co-occurrence matrix
	cooc_matrix: np.ndarray = np.zeros((n_gray_levels, n_gray_levels), dtype=np.int32)
	
	# Build co-occurrence matrix using vectorized operations
	i_indices, j_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
	ni: np.ndarray = i_indices + dy
	nj: np.ndarray = j_indices + dx
	
	# Create mask for valid indices
	valid_mask: np.ndarray = (ni >= 0) & (ni < height) & (nj >= 0) & (nj < width)
	
	# Get values for valid pairs and ensure integer type
	i_vals: np.ndarray = image[i_indices[valid_mask], j_indices[valid_mask]]
	j_vals: np.ndarray = image[ni[valid_mask], nj[valid_mask]]
	
	# Ensure values are within valid range before indexing
	i_vals = np.clip(i_vals.astype(np.int32), 0, n_gray_levels-1)
	j_vals = np.clip(j_vals.astype(np.int32), 0, n_gray_levels-1)
	
	# Update co-occurrence matrix using np.add.at for atomic operations
	np.add.at(cooc_matrix, (i_vals, j_vals), 1)
	
	# Normalize matrix
	return cooc_matrix / cooc_matrix.sum()

# Haralick
def haralick(img: ImageData, distances: list[int] = [1, 2, 3, 4], angles: list[int] = [0, 45, 90, 135, 180, 225, 270, 315]) -> ImageData:
	""" Compute Haralick texture features for an image using co-occurrence matrices.\n
	Args:
		img			(ImageData):	Input image (2D grayscale)
		distances	(list[int]):	List of distances to consider (default: [1,2,3,4])
		angles		(list[int]):	List of angles in degrees (default: [0,45,90,135,180,225,270,315])
	Returns:
		ImageData: Array of Haralick features (n_distances * n_angles * 4 metrics)
	"""
	# Ensure input is 2D
	if len(img.shape) != 2:
		images: list[np.ndarray] = [haralick(img[i], distances, angles).data for i in range(img.shape[0])]
		return ImageData(np.concatenate(images), "Haralick Features")
		
	image: np.ndarray = img.data
	
	# Parameters validation
	assert distances, "Distances list cannot be empty"
	assert angles, "Angles list cannot be empty"
	assert min(distances) >= 1, "Distances must be positive integers"
		
	n_gray_levels: int = 256
	features: list[float] = []
	
	# For each distance and angle
	for d in distances:
		for theta in angles:
			# Get co-occurrence matrix
			cooc_matrix: np.ndarray = _create_cooc_matrix(image, d, theta, n_gray_levels)
			
			# Calculate Haralick features
			# 1. Energy (Angular Second Moment)
			angular_second_moment: float = np.sum(cooc_matrix ** 2)
			
			# 2. Contrast
			indices: np.ndarray = np.indices((n_gray_levels, n_gray_levels))
			contrast: float = np.sum(((indices[0] - indices[1]) ** 2) * cooc_matrix)
			
			# 3. Correlation
			mu_i: float = np.sum(indices[0] * cooc_matrix)
			mu_j: float = np.sum(indices[1] * cooc_matrix)
			sigma_i: float = np.sqrt(np.sum(((indices[0] - mu_i) ** 2) * cooc_matrix))
			sigma_j: float = np.sqrt(np.sum(((indices[1] - mu_j) ** 2) * cooc_matrix))
			
			# Add small epsilon to avoid division by zero
			epsilon = 1e-10
			correlation: float = np.sum((indices[0] - mu_i) * (indices[1] - mu_j) * cooc_matrix) / (sigma_i * sigma_j + epsilon)
			
			# 4. Homogeneity (Inverse Difference Moment)
			homogeneity: float = np.sum(cooc_matrix / (1 + (indices[0] - indices[1]) ** 2))
			
			features.extend([angular_second_moment, contrast, correlation, homogeneity])
	
	return ImageData(np.array(features), "Haralick Features")


## Others
# CNN (VGG-16)
def cnn_vgg16(img: ImageData) -> ImageData:
	""" Compute the features of an image using a pre-trained CNN (VGG-16).\n
	Args:
		img		(ImageData):	Image
	Returns:
		ImageData: Array of features
	"""
	from keras.applications.vgg16 import VGG16
	import tensorflow as tf
	model: tf.keras.Model = VGG16(include_top=False, weights="imagenet")

	# Preprocess image for VGG16 (3 channels of size 224x224)
	if len(img.shape) == 2:
		# For grayscale, replicate to 3 channels
		img_data = np.stack((img.data,) * 3, axis=-1)
	else:
		# For RGB, transpose from (3, H, W) to (H, W, 3) and make sure it's 3 channels
		img_data = np.transpose(img.data, (1, 2, 0))[:, :, :3]
	img_data = tf.image.resize(img_data, (224, 224))
	img_data = np.expand_dims(img_data, axis=0)
	
	# Get features from last convolutional layer (flattened) and return them
	features: np.ndarray = model.predict(img_data, verbose=0)
	return ImageData(features.flatten(), "CNN Features")


# Name every function
from typing import Callable
DESCRIPTORS_CALLS: dict[str, Callable] = {
	# Histograms
	"Histogram":					{"function":histogram_multi_channels, "args":{}},
	"Histogram (HSV/HSL)":			{"function":histogram_hue_per_saturation, "args":{}},
	"Histogram Blob":				{"function":histogram_blob, "args":{}},

	# Formes
	# "Gradient Magnitude":			{"function":gradient_magnitude, "args":{}},
	# "Gradient Orientation":			{"function":gradient_orientation, "args":{}},
	"Weighted Gradient Histogram":	{"function":weighted_gradient_histogram, "args":{}},

	# Textures
	"Statistics":					{"function":statistics, "args":{}},
	"Local Binary Pattern":			{"function":local_binary_pattern, "args":{}},
	"Haralick":						{"function":haralick, "args":{}},

	# Others
	"CNN (VGG-16)":					{"function":cnn_vgg16, "args":{}},
}
