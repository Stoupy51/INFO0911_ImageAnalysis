
# Imports
from config import *
from src.processing.fastaniso import anisodiff, anisodiff3
from src.print import *
from PIL import Image
import numpy as np

# Constants
BEST_MODEL: str = "inverse_iter15_k100_g0.1"		# for anisodiff
NB_ITERATIONS: int =	int(BEST_MODEL.split("iter")[1].split("_")[0])
KAPPA: int =			int(BEST_MODEL.split("k")[1].split("_")[0])
GAMMA: float =			float(BEST_MODEL.split("g")[1].split("_")[0])
OPTION: int = 1 if BEST_MODEL.split("_")[-1] == "inverse" else 0
INPUT_IMAGES_FOLDER: str = f"{IMAGE_FOLDER}/to_process"		# Folder containing the images to process
COLOR_PALETTE: str = "L"	# Color palette for the output image to save

# Main function
def main():

	# Print the start time
	info(f"Start time: {START_TIME_STR}")

	# Get all the images in the folder
	if not os.path.exists(INPUT_IMAGES_FOLDER):
		error(f"Folder '{INPUT_IMAGES_FOLDER}' does not exist")
	images: list[str] = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.endswith((".jpg",".png"))]
	if not images:
		error(f"No images found in '{INPUT_IMAGES_FOLDER}'")
	os.makedirs(f"{OUTPUT_FOLDER}/{BEST_MODEL}", exist_ok=True)

	# For each image,
	for image_name in images:
		image_path: str = f"{INPUT_IMAGES_FOLDER}/{image_name}"
		output_path: str = f"{OUTPUT_FOLDER}/{BEST_MODEL}/{image_name.replace('.png','.jpg')}"

		# Load the image
		image: Image.Image = Image.open(image_path)
		image_array: np.ndarray = np.array(image)
		if image_array.ndim == 3:
			image_array = np.mean(image_array, axis=-1)

		# Apply the anisotropic diffusion
		output: np.ndarray = anisodiff(image_array, niter=NB_ITERATIONS, kappa=KAPPA, gamma=GAMMA, option=OPTION, ploton=False)

		# Save the output image
		output_image: Image.Image = Image.fromarray(output).convert(COLOR_PALETTE)
		output_image.save(output_path, format="JPEG", quality=JPG_QUALITY)
		debug(f"Image '{output_path}' saved successfully")

	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

