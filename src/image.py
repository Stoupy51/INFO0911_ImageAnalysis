
# Imports
from __future__ import annotations
import numpy as np

# TODO <--- USE THIS SHIT
class ImageData(object):
	
	def __init__(self, img: np.ndarray, color_space: str, channel: int|None = None):
		""" ImageData constructor\n
		Args:
			img			(np.ndarray):	Numpy 2D or 3D array containing the image
			color_space	(str):			Used color space, ex: "RGB", "HSV", "Indexation (8,8,8)"
			channel		(int|None):		Used for 2D image to precise why values_range to pick.
		"""
		self.data: np.ndarray = img
		self.color_space: str = color_space
		self.channel = channel

		# If not 3D and no channel provided, default to 0
		if channel is None and len(self.data.shape) < 3:
			self.channel = 0

	@property
	def values_ranges(self) -> list[tuple[float, float, float]]:
		""" Get the ranges of the values in the image depending on the color space\n
		Returns:
			list: for each channel, a tuple with the following format:
				minimum	(float):	Minimum value in the image
				maximum	(float):	Maximum value in the image + 1 (for the range)
				step	(float):	Step size for the values
		"""
		if self.color_space in ["YUV", "YIQ"]:
			return [(0, 256, 1), (0, 1, 0.1), (0, 1, 0.1)]
		elif "RGB" in self.color_space and "ormaliz" in self.color_space:
			return [(0, 1, 1/256)] * 3
		elif self.color_space in ["HSV", "HSL"]:
			return [(0, 360, 1), (0, 1, 0.1), (0, 1, 0.1)]
		elif self.color_space == "CMYK":
			return [(0, 1, 0.1)] * 4
		elif self.color_space == "L*a*b":
			return [(0, 100, 1), (-128, 128, 1), (-128, 128, 1)]
		elif self.color_space == "L*u*v":
			return [(0, 100, 1), (-134, 220, 1), (-140, 122, 1)]
		elif "Indexation" in self.color_space:
			# Extract '8,8,8' from 'Indexation (8,8,8)'
			maxi: list[str] = self.color_space.split('(')[1].split(')')[0].split(',')
			ranges = []
			for i in range(len(maxi)):
				ranges.append( (0, int(maxi[i]), 1) )
			return ranges

		# RGB like
		return [(0, 256, 1)] * self.data.shape[0]

	@property
	def range(self) -> tuple[float, float, float]:
		""" Get the range of the values in the image depending on the color space\n
		Returns:
			tuple: a tuple with the following format:
				minimum	(float):	Minimum value in the image
				maximum	(float):	Maximum value in the image + 1 (for the range)
				step	(float):	Step size for the values
		"""
		if self.channel is None:
			raise ValueError("ImageData.range: 3D image, please provide a channel or use ImageData.values_ranges")
		return self.values_ranges[self.channel]

	@property
	def shape(self) -> tuple[int, ...]:
		""" Get the shape of the image\n
		Returns:
			tuple: Shape of the image
		"""
		return self.data.shape

	# Get a specific channel from the image
	def __getitem__(self, i: int) -> ImageData:
		return ImageData(self.data[i], self.color_space, channel=i)

