from src.descriptors import *
from src.print import *

data: str = """
12, 13, 0, 255, 27, 1
58, 67, 52, 4, 3, 0
32, 100, 200, 5, 22, 0
43,45,67,210,255,220
12,45,70,59,20,30
10,40,7,58,3,50
15, 8, 9, 57, 56, 53
""".strip().replace(" ", "")

data: list[list[int]] = [list(map(int, row.split(","))) for row in data.split("\n")]
array: np.ndarray = np.array(data, dtype=np.uint8)
print(local_binary_pattern(ImageData(array, "RGB")).data)

# Big image test
img: ImageData = ImageData(np.random.randint(0, 256, (100, 100), dtype=np.uint8), "RGB")

@measure_time(debug)
def time_lbp_optimized():
    return local_binary_pattern(img, optimized=True)

@measure_time(debug)
def time_lbp_unoptimized():
    return local_binary_pattern(img, optimized=False)

optimized_lbp = time_lbp_optimized()
unoptimized_lbp = time_lbp_unoptimized()
print("LBP results are equal:", np.array_equal(optimized_lbp.data, unoptimized_lbp.data))

# Test histogram_blob

@measure_time(debug)
def time_optimized():
    return histogram_blob(img)

@measure_time(debug) 
def time_unoptimized():
    return histogram_blob(img, optimized=False)

optimized_result = time_optimized()
unoptimized_result = time_unoptimized()

print("Histogram blob results are equal:", np.array_equal(optimized_result.data, unoptimized_result.data))

