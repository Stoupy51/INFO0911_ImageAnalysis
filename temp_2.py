
import numpy as np
from scipy.ndimage import convolve
ROUND_PRINT: int = 1

values: str = """
0,        0,    0, 	255, 255, 255
0,        0,    0, 	255, 255, 255
0,        0,    0, 	255, 255, 255
255, 255, 255, 0,      0,      0
255, 255, 255, 0,      0,      0
255, 255, 255, 0,      0,      0
""".replace(" ", "").replace('\t','').strip()
array: np.ndarray = np.array([[int(j) for j in i.split(",")] for i in values.split("\n")])
print(repr(array))


filters: np.ndarray = np.array([
	[
		[-1, 0, 1],
		[-1, 0, 1],
		[-1, 0, 1],
	],
	[
		[1, 1, 1],
		[0, 0, 0],
		[-1, -1, -1],
	],
])
results: list[np.ndarray] = []

for f in filters:
	print(repr(f))

	# Apply filter
	applied: np.ndarray = convolve(array, f)[1:-1, 1:-1] / (f.shape[0] * f.shape[1])
	results.append(applied)
	print(np.round(applied, ROUND_PRINT))
	print()


dx, dy = results
norm = np.sqrt(dx**2 + dy**2)
print(np.round(norm, ROUND_PRINT))

orientation = np.arctan2(dy, dx) * 180 / np.pi
orientation = np.where(dx == 0, 0, orientation)
orientation = np.where(dy == 0, 0, orientation)
print(np.round(orientation, ROUND_PRINT))


