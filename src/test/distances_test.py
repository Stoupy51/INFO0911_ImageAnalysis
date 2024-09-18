
# Imports
from src.vector_utils import *
from src.distances import *
from src.print import *
from typing import Callable
import time

# Test function
def test_all():
	""" Test all the functions """
	# Two known vectors for other tests
	v1: np.ndarray = np.array([2.7, 4.3,  0.2,   9,   -4])
	v2: np.ndarray = np.array([7.6, 5.8, -3.2, 9.7, 12.3])

	# Generate two random vectors
	v3: np.ndarray = random_vector(10)
	v4: np.ndarray = random_vector(10)

	# Similar vectors
	v5: np.ndarray = np.array([1, 2, 3, 4, 5])
	v6: np.ndarray = np.array([1, 2, 4, 4, 5])

	# Dictionnary test and inputs
	tests: dict = {
		"Manhattan":				{"function":distance_manhattan},
		"Euclidean":				{"function":distance_euclidean},
		"Tchebyshev":				{"function":distance_tchebyshev},
		"Minkowski":				{"function":distance_minkowski},
		"Histogram Intersection":	{"function":distance_histogram_intersection},
		"Khi2":						{"function":distance_khi2},
	}
	inputs: list[tuple] = [
		("v1/v2 (vecteurs du cours)", v1, v2),
		("v3/v4 (vecteurs aléatoires)", v3, v4),
		("v5/v6 (vecteurs similaires)", v5, v6),
	]

	# Test all the inputs
	for input_type, x, y in inputs:
		for test in tests.values():
			f: Callable = test["function"]
			
			# Measure the time
			start: int = time.perf_counter_ns()
			distance: float = f(x,y)
			end: int = time.perf_counter_ns()
			duration: int = end - start

			# Store the result
			test[input_type] = {"distance":distance, "duration":duration}
		
		# Try Histogram Intersection but with x and y swapped
		f: Callable = distance_histogram_intersection
		start: int = time.perf_counter_ns()
		distance: float = f(y,x)
		end: int = time.perf_counter_ns()
		duration: int = end - start
		tests["Histogram Intersection"][input_type] = {"distance":distance, "duration":duration}
	
	# Print the results
	for test_name, test in tests.items():
		info(f"Test {test_name}")
		for input_type, result in test.items():
			if input_type == "function":
				continue
			distance: float = round(result["distance"], 5)
			duration: int = result["duration"]
			info(f"	{input_type}: {distance} in {duration} ns")

	# TODO: transformer en tableau (les fonctions sur l'horizontale, le result sur la verticale)

