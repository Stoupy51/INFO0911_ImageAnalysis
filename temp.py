
import numpy as np

values: str = """
2,3,4,155,160,141,250
132, 14, 27, 0, 255, 100, 19
21,72,10,33,23,47,58
29, 0, 180, 256, 15, 7, 8
3,160,221,19,0,0,1
4,21,23,22,70,79,58
7, 21, 32,19,21,21,19
48,47,45,53,120,137,151
"""
array: np.ndarray = np.array([[int(j) for j in i.split(",")] for i in values.strip().split("\n")])

# Compute gradient
gradient_x: np.ndarray = np.zeros((array.shape[0], array.shape[1] - 1))
for i in range(array.shape[0]):
	for j in range(array.shape[1] - 1):
		gradient_x[i, j] = (array[i, j+1] - array[i, j])
print(gradient_x)
print()

gradient_y: np.ndarray = np.zeros((array.shape[0] - 1, array.shape[1]))
for i in range(array.shape[0] - 1):
	for j in range(array.shape[1]):
		gradient_y[i, j] = (array[i, j] - array[i+1, j])
print(gradient_y)
print()

test = np.array([[0]*3 + [255] * 3] * 6)
print(test)
print(np.gradient(test)[1])

