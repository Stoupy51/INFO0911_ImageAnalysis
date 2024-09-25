
# Imports
from config import *
from src.processing.fastaniso import anisodiff, anisodiff3
from src.print import *
from PIL import Image
import numpy as np
import pydicom as dicom

# Constants
INPUT_IMAGES_FOLDER: str = f"{IMAGE_FOLDER}/dicom"				# Folder containing the images to process
OUTPUT_IMAGES_FOLDER: str = f"{OUTPUT_FOLDER}/dicom_to_jpg"		# Folder containing the images to process
COLOR_PALETTE: str = "L"										# Color palette for the output image to save

# Main function
def main():

	# Print the start time
	info(f"Start time: {START_TIME_STR}")

	# Get all the images in the folder
	if not os.path.exists(INPUT_IMAGES_FOLDER):
		error(f"Folder '{INPUT_IMAGES_FOLDER}' does not exist")
	images: list[str] = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.endswith((".dicom"))]
	if not images:
		error(f"No images found in '{INPUT_IMAGES_FOLDER}'")

	# For each image,
	for image_name in images:
		image_path: str = f"{INPUT_IMAGES_FOLDER}/{image_name}"
		output_path: str = f"{OUTPUT_IMAGES_FOLDER}/{image_name.replace('.dicom','')}"
		os.makedirs(output_path, exist_ok=True)

		# Load the dicom image
		dicom_image: dicom.dataset.FileDataset = dicom.read_file(image_path)

		# Save all the images in the folder
		for i, image in enumerate(dicom_image.pixel_array):
			output_image: Image.Image = Image.fromarray(image).convert(COLOR_PALETTE)
			output_image.save(f"{output_path}/{i}.jpg", format="JPEG", quality=JPG_QUALITY)
			debug(f"Image '{output_path}/{i}.jpg' saved successfully")

	
	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

