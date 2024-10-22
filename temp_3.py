
import numpy as np
from scipy.ndimage import convolve
import cv2
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
	#applied = cv2.filter2D(array, -1, f)
	results.append(applied)
	print(np.round(applied, ROUND_PRINT))
	print()


dx, dy = results
norm = np.sqrt(dx**2 + dy**2)
print(np.round(norm, ROUND_PRINT))

import time

st = time.perf_counter_ns()
orientation = (np.arctan2(dy, dx) * 180 / np.pi) % 360
orientation = np.where(dx == 0, 0, orientation)
orientation = np.where(dy == 0, 0, orientation)
ed = time.perf_counter_ns()
print(ed - st)
print(np.round(orientation, ROUND_PRINT))

st = time.perf_counter_ns()
orientation_v2 = cv2.phase(dx, dy, angleInDegrees=True)
orientation_v2 = np.where(dx == 0, 0, orientation_v2)
orientation_v2 = np.where(dy == 0, 0, orientation_v2)
ed = time.perf_counter_ns()
print(ed - st)
print(np.round(orientation_v2, ROUND_PRINT))

