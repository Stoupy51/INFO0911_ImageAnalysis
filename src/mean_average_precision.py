
# Import config from the parent folder
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
from print import *

# Imports
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from src.search_engine import search, NORMALIZATION_CALLS
from tqdm import tqdm

# Functions
def get_class_from_path(image_path: str) -> str:
	""" Get the class name from the image path\n
	Args:
		image_path	(str):	Path to the image
	Returns:
		str: Class name
	"""
	return image_path.replace("\\","/").split("/")[-2]

def get_all_images() -> dict[str, list[str]]:
	""" Get all images paths grouped by class\n
	Returns:
		dict[str, list[str]]: Dictionary of class names and their image paths
	"""
	images_by_class = defaultdict(list)
	for root, _, files in os.walk(DATABASE_FOLDER):
		for file in files:
			if file.endswith(IMAGE_EXTENSIONS):
				path = f"{root}/{file}"
				class_name = get_class_from_path(path)
				images_by_class[class_name].append(path)
	return dict(images_by_class)

def compute_precision_recall(relevant_ranks: list[int], total_relevant: int, total_results: int) -> tuple[list[float], list[float]]:
	""" Compute precision and recall values at each rank\n
	Args:
		relevant_ranks	(list[int]):	List of ranks of relevant images
		total_relevant	(int):		Total number of relevant images
		total_results	(int):		Total number of results
	Returns:
		tuple[list[float], list[float]]: Precision and recall values
	"""
	precisions: list[float] = []
	recalls: list[float] = []
	found_relevant: int = 0
	
	for rank in range(total_results):
		if rank in relevant_ranks:
			found_relevant += 1
		precision = found_relevant / (rank + 1)
		recall = found_relevant / total_relevant
		precisions.append(precision)
		recalls.append(recall)
		
	return precisions, recalls

def compute_average_precision(relevant_ranks: list[int], k: int|None = None) -> float:
	""" Compute Average Precision score\n
	Args:
		relevant_ranks	(list[int]):	List of ranks of relevant images
		k				(int|None):	Optional k value for MAP@K
	Returns:
		float: Average Precision score
	"""
	if not relevant_ranks:
		return 0.0
		
	precisions: list[float] = []
	num_relevant: int = 0
	
	for i, rank in enumerate(sorted(relevant_ranks)):
		if k is not None and rank >= k:
			continue
		num_relevant += 1
		precision = num_relevant / (rank + 1)
		precisions.append(precision)
	
	if not precisions:
		return 0.0
		
	return sum(precisions) / len(precisions)

@handle_error(message="Error during evaluation", error_log=2)
def evaluate_descriptors(color_spaces: list[str], descriptors: list[str], 
						distances: list[str], k: int|None = None,
						plot_curves: bool = False) -> dict[str, dict[str, float]]:
	""" Evaluate descriptors using MAP and MAP@K\n
	Args:
		color_spaces	(list[str]):	List of color spaces to evaluate
		descriptors	(list[str]):	List of descriptors to evaluate
		distances	(list[str]):	List of distances to evaluate
		k			(int|None):	Optional k value for MAP@K
		plot_curves	(bool):		Whether to plot precision-recall curves
	Returns:
		dict[str, dict[str, float]]: Results with format {descriptor: {distance: map_score}}
	"""
	# Get all images grouped by class
	images_by_class: dict = get_all_images()
	total_images: int = sum(len(imgs) for imgs in images_by_class.values())
	
	# Store results and curves data
	results = defaultdict(lambda: defaultdict(float))
	curves_data = defaultdict(lambda: defaultdict(list)) if plot_curves else None
	
	# Get normalization
	normalization: str = list(NORMALIZATION_CALLS.keys())[0]
	
	# For each image as query
	total_queries: int = sum(len(class_images) for class_images in images_by_class.values())
	with tqdm(total=total_queries, desc="Evaluating queries") as pbar:
		for class_name, class_images in images_by_class.items():
			for query_path in class_images:
				# Get query image
				query_image: Image.Image = Image.open(query_path).convert("RGB")
				
				# Get remaining images as candidates (leave-one-out)
				other_relevant: list[str] = [p for p in class_images if p != query_path]
				total_relevant: int = len(other_relevant)
				
				# For each descriptor+distance combination
				for cs in color_spaces:
					for desc in descriptors:
						key: str = f"{cs}_{desc}"
						try:
							for dist in distances:
								# Search similar images
								results_list = search(query_image, [cs], [desc], normalization, dist, 
												max_results=total_images-1)
								
								# Get ranks of relevant images (same class)
								relevant_ranks: list[int] = []
								for rank, (path, _, _) in enumerate(results_list):
									if get_class_from_path(path) == class_name:
										relevant_ranks.append(rank)
										
								# Compute AP and add to results
								ap: float = compute_average_precision(relevant_ranks, k)
								results[key][dist] += ap / total_images
								
								# Store precision-recall curve data if requested
								if plot_curves:
									precisions, recalls = compute_precision_recall(
										relevant_ranks, total_relevant, len(results_list))
									curves_data[key][dist].append((precisions, recalls))
						except Exception as e:
							warning(f"Error evaluating {cs}_{desc}: {str(e)}")
							continue
				pbar.update(1)
	
	# Plot precision-recall curves if requested
	if plot_curves:
		plot_precision_recall_curves(curves_data)
		
	return dict(results)

def plot_precision_recall_curves(curves_data: dict[str, dict[str, list[tuple[list[float], list[float]]]]]) -> None:
	""" Plot precision-recall curves for each descriptor+distance combination\n
	Args:
		curves_data (dict): Curves data with format {descriptor: {distance: [(precisions, recalls)]}}
	"""
	plt.figure(figsize=(12, 8))
	
	for desc_key, distances in curves_data.items():
		for dist_key, curves in distances.items():
			# Average curves across all queries
			avg_precision = np.mean([p for p, _ in curves], axis=0)
			avg_recall = np.mean([r for _, r in curves], axis=0)
			
			plt.plot(avg_recall, avg_precision, label=f"{desc_key}-{dist_key}")
	
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall Curves")
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("precision_recall_curves.png")
	plt.close()

def print_evaluation_results(results: dict[str, dict[str, float]]) -> None:
	""" Print evaluation results in a formatted table\n
	Args:
		results (dict): Results with format {descriptor: {distance: map_score}}
	"""
	# Get all distances
	distances: set = set()
	for desc_results in results.values():
		distances.update(desc_results.keys())
	distances = sorted(distances)
	
	# Print header
	print("\nEvaluation Results:")
	header: str = "Descriptor".ljust(30) + " | " + " | ".join(d.ljust(10) for d in distances)
	print("-" * len(header))
	print(header)
	print("-" * len(header))
	
	# Print results for each descriptor
	for desc, desc_results in sorted(results.items()):
		row: str = desc.ljust(30) + " | "
		row += " | ".join(f"{desc_results.get(d, 0):.4f}".ljust(10) for d in distances)
		print(row)
	print("-" * len(header))
