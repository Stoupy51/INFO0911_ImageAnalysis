
# Imports
from config import *
from src.print import *
from src.dicom import dicom_to_jpg

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
		dicom_to_jpg(image_path, output_path, color_palette=COLOR_PALETTE)
	
	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

