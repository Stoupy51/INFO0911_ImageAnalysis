
# Imports
import os
import pydicom
import numpy as np
from PIL import Image
from config import JPG_QUALITY
from src.print import *

# Function to convert DICOM to JPG
def dicom_to_jpg(dicom_file: str, output_dir: str, quality: int = JPG_QUALITY, color_palette: str = "L"):
	""" Convert DICOM file to JPG image(s)\n
	Args:
		dicom_file		(str):	Path to the DICOM file
		output_dir		(str):	Directory to save the JPG file(s)
		quality			(int):	Quality of the JPG file (0-100)
		color_palette	(str):	Color palette for the output image to save
	"""
	os.makedirs(output_dir, exist_ok=True)

	# Read the DICOM file
	ds: pydicom.FileDataset = pydicom.dcmread(dicom_file)

	# Check if the DICOM file contains multiple frames
	if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
		for i in range(ds.NumberOfFrames):
			# Extract the image data for each frame
			image_data = ds.pixel_array[i]

			# Convert the image data to a PIL Image
			img = Image.fromarray(image_data)

			# Normalize the image to 8-bit (0-255)
			img = img.convert(color_palette)  # Convert to grayscale or other color palette
			img = img.point(lambda x: x * 255.0 / np.max(img))  # Normalize

			# Save the image as JPG
			output_file = f"{output_dir}/frame_{i + 1}.jpg"
			img.save(output_file, quality=quality)
			info(f"Saved: {output_file}")
	else:
		# Handle single frame DICOM
		image_data = ds.pixel_array
		img = Image.fromarray(image_data)
		img = img.convert(color_palette)  # Convert to grayscale or other color palette
		img = img.point(lambda x: x * 255.0 / np.max(img))  # Normalize

		output_file = f"{output_dir}/frame_1.jpg"
		img.save(output_file, quality=quality)
		info(f"Saved: {output_file}")

