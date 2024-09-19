
# Imports
from main import *
from distances import *

if __name__ == "__main__":

	# Get flatten original image
	flatten_original: np.ndarray = BASE_IMAGE.flatten()

	# For each folder,
	for noise in NOISES:

		# Sort each image by their distance to the original
		images: list[str] = os.listdir(f"{IMAGE_NAME}/{noise}")
		distance_per_image: dict[str, float] = {}

		# For each image,
		for image_path in images:

			# Load image and convert as np array
			image: np.ndarray = np.array(Image.open(f"{IMAGE_NAME}/{noise}/{image_path}").convert("L")).flatten()

			# Calculate distance
			distance_per_image[image_path] = distance_euclidean(flatten_original, image)

		# Sort images by distance
		sorted_images: list[str] = sorted(distance_per_image, key=distance_per_image.get)

		# Print top 3
		print(f"Top 3 images for noise {noise}:")
		for i in range(3):
			print(f"\t{IMAGE_NAME}/{noise}/{sorted_images[i]}: {distance_per_image[sorted_images[i]]}")

