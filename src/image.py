
# Imports
import numpy as np


# TODO <--- USE THIS SHIT
class ImageData(object):
	
	def __init__(self, img: np.ndarray, color_space: str):
		self.data: np.ndarray = img
		self.color_space: str = color_space

	@property
	def values_ranges(self) -> list[tuple[float, float, float]]:
		""" Get the ranges of the values in the image depending on the color space\n
		Returns:
			list: for each channel, a tuple with the following format:
				minimum	(float):	Minimum value in the image
				maximum	(float):	Maximum value in the image + 1 (for the range)
				step	(float):	Step size for the values
		"""
		values_ranges: list[tuple[float,float,float]] = [(0, 256, 1)] * 3
		if self.color_space in ["YUV", "YIQ"]:
			values_ranges = [(0, 256, 1), (0, 1, 0.1), (0, 1, 0.1)]
		elif self.color_space in ["HSV", "HSL"]:
			values_ranges = [(0, 360, 1), (0, 1, 0.1), (0, 1, 0.1)]
		elif self.color_space == "CMYK":
			values_ranges = [(0, 1, 0.1)] * 4
		elif self.color_space == "L*a*b":
			values_ranges = [(0, 100, 1), (-128, 128, 1), (-128, 128, 1)]
		elif self.color_space == "L*u*v":
			values_ranges = [(0, 100, 1), (-134, 220, 1), (-140, 122, 1)]
		elif "Indexation" in self.color_space:
			# Get the maximum values from the color space string 'Indexation (8,8,8)'
			maxi: list[str] = self.color_space.split('(')[1].split(')')[0].split(',')
			values_ranges = []
			for i in range(len(maxi)):
				values_ranges.append((0, int(maxi[i]), 1))

		# If 2D, return only the first range
		if len(self.data.shape) == 2:
			values_ranges = [values_ranges[0]]

		return values_ranges

	# TODO: color_space[i]
	def __getitem__(self, i: int) -> "ImageData":
		return ImageData(self.data[i], self.color_space)

