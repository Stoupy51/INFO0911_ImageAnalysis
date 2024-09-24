
# Imports
import numpy as np
import cv2

# Usable function
def cedf_filter(image_2d: np.ndarray|str, alpha: float = 0.001, sigma: float = 0.7, rho: float = 4.0, C: float = 1, T: float = 15, step_t: float = 0.15) -> np.ndarray:
	""" Coherence-Enhancing-Diffusion-filtering algorithm\n
	Args:
		image_2d	(np.ndarray|str):	Grayscaled input image (2D array)
		alpha		(float):			Alpha value (used for calculation of diffusion matrix)
		sigma		(float):			Standard deviation for gaussian kernel
		rho			(float):			Standard deviation for gaussian kernel
		T			(float):			Maximum value of steps (nb of iterations depends on step_t)
		C			(float):			Constant value for calculation of diffusion matrix
		step_t		(float):			Step value for iteration
	Returns:
		(np.ndarray): Filtered image (2D array)
	"""
	# Prepare variables
	current_step: float = 0.0
	if isinstance(image_2d, str):
		converted_image: np.ndarray = cv2.imread(image_2d, cv2.IMREAD_GRAYSCALE).astype(np.float64)
	else:
		converted_image: np.ndarray = image_2d.astype(np.float64)
	nb_rows, nb_cols = converted_image.shape

	# Main loop
	while current_step < (T - 0.001):
		current_step += step_t
		print(current_step)

		# Gaussian kernel (with sigma)
		limit_x: np.ndarray = np.arange(-np.ceil(2 * sigma), np.ceil(2 * sigma) + 1)
		k_sigma: np.ndarray = np.exp(-(limit_x**2 / (2 * sigma**2)))
		k_sigma /= np.sum(k_sigma)
		u_sigma: np.ndarray = cv2.filter2D(cv2.filter2D(converted_image, -1, k_sigma[:, np.newaxis]), -1, k_sigma)

		# Gradient
		uy, ux = np.gradient(u_sigma)
		
		# Apply the Gaussian kernel with rho to smooth gradient components
		limit_x_j: np.ndarray = np.arange(-np.ceil(3 * rho), np.ceil(3 * rho) + 1)
		k_sigma_rho: np.ndarray = np.exp(-(limit_x_j**2 / (2 * rho**2)))
		k_sigma_rho /= np.sum(k_sigma_rho)
		Jxx: np.ndarray = cv2.filter2D(cv2.filter2D(ux**2, -1, k_sigma_rho[:, np.newaxis]), -1, k_sigma_rho)
		Jxy: np.ndarray = cv2.filter2D(cv2.filter2D(ux * uy, -1, k_sigma_rho[:, np.newaxis]), -1, k_sigma_rho)
		Jyy: np.ndarray = cv2.filter2D(cv2.filter2D(uy**2, -1, k_sigma_rho[:, np.newaxis]), -1, k_sigma_rho)

		# Prepare arrays for principal axis transformation
		v2x: np.ndarray = np.zeros((nb_rows, nb_cols), dtype=np.float64)
		v2y: np.ndarray = np.zeros((nb_rows, nb_cols), dtype=np.float64)
		lambda1: np.ndarray = np.zeros((nb_rows, nb_cols), dtype=np.float64)
		lambda2: np.ndarray = np.zeros((nb_rows, nb_cols), dtype=np.float64)

		# Perform eigenvalue decomposition for each pixel
		for i in range(nb_rows):
			for j in range(nb_cols):
				J_matrix: np.ndarray = np.array([[Jxx[i, j], Jxy[i, j]], [Jxy[i, j], Jyy[i, j]]], dtype=np.float64)
				eig_vals, eig_vecs = np.linalg.eig(J_matrix)
				lambda1[i, j], lambda2[i, j] = eig_vals
				v2x[i, j], v2y[i, j] = eig_vecs[:, 1]

				# Normalize eigenvector
				norm: float = np.sqrt(v2x[i, j]**2 + v2y[i, j]**2)
				if norm > 0:
					v2x[i, j] /= norm
					v2y[i, j] /= norm

		# Calculate orthogonal vectors
		v1x: np.ndarray = -v2y
		v1y: np.ndarray = v2x

		# Calculate diffusion matrix components
		delta_lambda: np.ndarray = lambda1 - lambda2
		lambda1_mod: np.ndarray = alpha + (1 - alpha) * np.exp(-C / (delta_lambda**2 + 1e-12))  # Add epsilon for numerical stability
		lambda2_mod: np.ndarray = alpha

		Dxx: np.ndarray = lambda1_mod * v1x**2 + lambda2_mod * v2x**2
		Dxy: np.ndarray = lambda1_mod * v1x * v1y + lambda2_mod * v2x * v2y
		Dyy: np.ndarray = lambda1_mod * v1y**2 + lambda2_mod * v2y**2

		# Non-negativity discretization update for the image
		converted_image = non_negativity_discretization(converted_image, Dxx, Dxy, Dyy, step_t)

	return converted_image.astype(np.uint8)


def non_negativity_discretization(im: np.ndarray, Dxx: np.ndarray, Dxy: np.ndarray, Dyy: np.ndarray, step_t: float) -> np.ndarray:
	"""Applies non-negativity discretization to update the image.\n
	Args:
		im		(np.ndarray):	The input image.
		Dxx		(np.ndarray):	The Dxx diffusion matrix component.
		Dxy		(np.ndarray):	The Dxy diffusion matrix component.
		Dyy		(np.ndarray):	The Dyy diffusion matrix component.
		step_t	(float):		The time step for updating the image.
	Returns:
		np.ndarray: The updated image after applying the discretization scheme.
	"""
	nb_rows, nb_cols = im.shape

	# Neighboring indices
	px: np.ndarray = np.concatenate(([1], np.arange(1, nb_rows)))
	nx: np.ndarray = np.concatenate((np.arange(1, nb_rows), [nb_rows - 1]))
	py: np.ndarray = np.concatenate(([1], np.arange(1, nb_cols)))
	ny: np.ndarray = np.concatenate((np.arange(1, nb_cols), [nb_cols - 1]))

	# Stencil weights calculation
	wbR1: np.ndarray = 0.25 * ((np.abs(Dxy[nx, py]) - Dxy[nx, py]) + (np.abs(Dxy) - Dxy))
	wtM2: np.ndarray = 0.5 * ((Dyy[:, py] + Dyy) - (np.abs(Dxy[:, py]) + np.abs(Dxy)))
	wbL3: np.ndarray = 0.25 * ((np.abs(Dxy[px, py]) + Dxy[px, py]) + (np.abs(Dxy) + Dxy))
	wmR4: np.ndarray = 0.5 * ((Dxx[nx, :] + Dxx) - (np.abs(Dxy[nx, :]) + np.abs(Dxy)))
	wmL6: np.ndarray = 0.5 * ((Dxx[px, :] + Dxx) - (np.abs(Dxy[px, :]) + np.abs(Dxy)))
	wtR7: np.ndarray = 0.25 * ((np.abs(Dxy[nx, ny]) + Dxy[nx, ny]) + (np.abs(Dxy) + Dxy))
	wmB8: np.ndarray = 0.5 * ((Dyy[:, ny] + Dyy) - (np.abs(Dxy[:, ny]) + np.abs(Dxy)))
	wtL9: np.ndarray = 0.25 * ((np.abs(Dxy[px, ny]) - Dxy[px, ny]) + (np.abs(Dxy) - Dxy))

	# Update the image using the stencil weights
	im += step_t * (wbR1 * (im[nx, py] - im) + wtM2 * (im[:, py] - im) +
					wbL3 * (im[px, py] - im) + wmR4 * (im[nx, :] - im) +
					wmL6 * (im[px, :] - im) + wtR7 * (im[nx, ny] - im) +
					wmB8 * (im[:, ny] - im) + wtL9 * (im[px, ny] - im))

	return im
