
# Imports
from config import *
from src.processing.fastaniso import anisodiff
from src.print import *
from PIL import Image
import numpy as np

# Constants
BEST_MODEL: str = "inverse_iter15_k100_g0.1"		# for anisodiff
NB_ITERATIONS: int =	int(BEST_MODEL.split("iter")[1].split("_")[0])
KAPPA: int =			int(BEST_MODEL.split("k")[1].split("_")[0])
GAMMA: float =			float(BEST_MODEL.split("g")[1].split("_")[0])
OPTION: int = 1 if BEST_MODEL.split("_")[-1] == "inverse" else 0
INPUT_IMAGES_FOLDER: str = f"{IMAGE_FOLDER}/to_apply"

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

	# For each image,
	for image_name in images:
		image_path: str = f"{INPUT_IMAGES_FOLDER}/{image_name}"
		output_path: str = f"{OUTPUT_FOLDER}/{BEST_MODEL}/{image_name}"
		os.makedirs(output_path, exist_ok=True)

		# Load the image
		image: Image.Image = Image.open(image_path)
		image_array: np.ndarray = np.array(image)

		# Apply the anisotropic diffusion and save the result
		output: np.ndarray = anisodiff(image, niter=NB_ITERATIONS, kappa=KAPPA, gamma=GAMMA, option=OPTION, ploton=False)
		output_image: Image.Image = Image.fromarray(output)
		output_image.save(output_path)

	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

