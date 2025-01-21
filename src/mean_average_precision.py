
# Import config from the parent folder
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
from print import *

# Imports
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.search_engine import search, NORMALIZATION_CALLS
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

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

def process_query(args):
	query_path, class_name, class_images, color_spaces, descriptors, distances, normalizations, k, total_images = args
	# Use regular dictionaries instead of defaultdict
	results = {}
	curves_data = {}
	
	# Get remaining images as candidates (leave-one-out)
	other_relevant: list[str] = [p for p in class_images if p != query_path]
	total_relevant: int = len(other_relevant)
	
	# For each descriptor+distance combination
	for cs in color_spaces:
		for desc in descriptors:
			key: str = f"{cs}_{desc}"
			try:
				# Initialize nested dictionaries
				if key not in results:
					results[key] = {}
					curves_data[key] = {}
				
				for dist in distances:
					if dist not in results[key]:
						results[key][dist] = {}
						curves_data[key][dist] = {}
					
					for norm in normalizations:
						# Search similar images
						results_list = search(query_path, [cs], [desc], norm, dist, 
										max_results=total_images-1, parallel="thread")
						
						# Get ranks of relevant images (same class)
						relevant_ranks: list[int] = []
						for rank, (path, _, _) in enumerate(results_list):
							if get_class_from_path(path) == class_name:
								relevant_ranks.append(rank)
								
						# Compute AP and add to results
						ap: float = compute_average_precision(relevant_ranks, k)
						results[key][dist][norm] = ap / total_images
						
						# Store precision-recall curve data
						precisions, recalls = compute_precision_recall(
							relevant_ranks, total_relevant, len(results_list))
						
						if norm not in curves_data[key][dist]:
							curves_data[key][dist][norm] = []
						curves_data[key][dist][norm].append((precisions, recalls))
			except Exception as e:
				import traceback
				warning(f"Error evaluating {cs}_{desc}: {str(e)}\n{traceback.format_exc()}")
				continue
	
	return dict(results), dict(curves_data)

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
	# Load existing results from CSV
	results_dir: str = "evaluation_results"
	os.makedirs(results_dir, exist_ok=True)
	results_file: str = f"{results_dir}/results.csv"
	
	if os.path.exists(results_file):
		results_df = pd.read_csv(results_file)
	else:
		results_df = pd.DataFrame(columns=[
			'color_space', 'descriptor', 'distance', 
			'normalization', 'k', 'map_score'
		])
	
	# Get all images grouped by class
	images_by_class: dict = get_all_images()
	total_images: int = sum(len(imgs) for imgs in images_by_class.values())
	
	# Get normalization methods to evaluate
	normalizations = list(NORMALIZATION_CALLS.keys())
	
	# Prepare arguments for multiprocessing
	args_list = []
	final_results = {}
	all_curves_data = {}
	
	# Check which combinations need to be evaluated
	for cs in color_spaces:
		for desc in descriptors:
			for dist in distances:
				for norm in normalizations:
					# Check if this combination already exists
					existing = results_df[
						(results_df['color_space'] == cs) &
						(results_df['descriptor'] == desc) &
						(results_df['distance'] == dist) &
						(results_df['normalization'] == norm) &
						(results_df['k'] == (k if k is not None else -1))
					]
					
					if len(existing) == 0:
						# Add to args list for evaluation
						for class_name, class_images in images_by_class.items():
							for query_path in class_images:
								args_list.append((
									query_path, class_name, class_images,
									[cs], [desc], [dist], [norm],
									k, total_images
								))
					else:
						# Use existing result
						desc_key = f"{cs}_{desc}"
						if desc_key not in final_results:
							final_results[desc_key] = {}
						if dist not in final_results[desc_key]:
							final_results[desc_key][dist] = {}
						final_results[desc_key][dist][norm] = existing.iloc[0]['map_score']
	
	# Process queries in parallel if there are new combinations to evaluate
	try:
		if args_list:
			with Pool(cpu_count()) as pool:
				for results, curves_data in tqdm(pool.imap_unordered(process_query, args_list), 
											total=len(args_list), desc="Evaluating queries"):
					if results:
						# Aggregate results
						for desc_key, distances_dict in results.items():
							if desc_key not in final_results:
								final_results[desc_key] = {}
							
							for dist_key, norm_dict in distances_dict.items():
								if dist_key not in final_results[desc_key]:
									final_results[desc_key][dist_key] = {}
								
								for norm_key, score in norm_dict.items():
									if norm_key not in final_results[desc_key][dist_key]:
										final_results[desc_key][dist_key][norm_key] = 0.0
									final_results[desc_key][dist_key][norm_key] += score
					
					# Aggregate curves data if plotting
					if plot_curves and curves_data:
						for desc_key, distances_dict in curves_data.items():
							if desc_key not in all_curves_data:
								all_curves_data[desc_key] = {}
							
							for dist_key, norm_dict in distances_dict.items():
								if dist_key not in all_curves_data[desc_key]:
									all_curves_data[desc_key][dist_key] = {}
								
								for norm_key, curves in norm_dict.items():
									if norm_key not in all_curves_data[desc_key][dist_key]:
										all_curves_data[desc_key][dist_key][norm_key] = []
									all_curves_data[desc_key][dist_key][norm_key].extend(curves)
			
			# Save aggregated results to CSV
			for desc_key, distances_dict in final_results.items():
				cs, desc = desc_key.split('_', 1)
				for dist_key, norm_dict in distances_dict.items():
					for norm_key, total_score in norm_dict.items():
						# Only save if this is a new combination
						if len(results_df[
							(results_df['color_space'] == cs) &
							(results_df['descriptor'] == desc) &
							(results_df['distance'] == dist_key) &
							(results_df['normalization'] == norm_key) &
							(results_df['k'] == (k if k is not None else -1))
						]) == 0:
							new_row = {
								'color_space': cs,
								'descriptor': desc,
								'distance': dist_key,
								'normalization': norm_key,
								'k': k if k is not None else -1,
								'map_score': total_score
							}
							results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
	
	except KeyboardInterrupt:
		warning("Evaluation interrupted by user")
	
	# Save updated results to CSV
	results_df.to_csv(results_file, index=False)
	
	# Plot precision-recall curves if requested
	if plot_curves:
		plot_precision_recall_curves(dict(all_curves_data))
		
	return dict(final_results)

def plot_precision_recall_curves(curves_data: dict[str, dict[str, dict[str, list[tuple[list[float], list[float]]]]]]) -> None:
	""" Plot precision-recall curves for each descriptor+distance+normalization combination\n
	Args:
		curves_data (dict): Curves data with format {descriptor: {distance: {normalization: [(precisions, recalls)]}}}
	"""
	plt.figure(figsize=(12, 8))
	
	for desc_key, distances in curves_data.items():
		for dist_key, normalizations in distances.items():
			for norm_key, curves in normalizations.items():
				# Average curves across all queries
				avg_precision = np.mean([p for p, _ in curves], axis=0)
				avg_recall = np.mean([r for _, r in curves], axis=0)
				
				plt.plot(avg_recall, avg_precision, label=f"{desc_key}-{dist_key}-{norm_key}")
	
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall Curves")
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("precision_recall_curves.png")
	plt.close()

def print_evaluation_results(results: dict[str, dict[str, dict[str, float]]]) -> None:
	""" Print evaluation results in a formatted table\n
	Args:
		results (dict): Results with format {descriptor: {distance: {normalization: map_score}}}
	"""
	# Get all distances and normalizations
	distances: set = set()
	normalizations: set = set()
	for desc_results in results.values():
		for dist_results in desc_results.values():
			normalizations.update(dist_results.keys())
		distances.update(desc_results.keys())
	distances = sorted(distances)
	normalizations = sorted(normalizations)
	
	# Print header
	print("\nEvaluation Results:")
	for norm in normalizations:
		header: str = f"\nNormalization: {norm}"
		print(header)
		header: str = "Descriptor".ljust(30) + " | " + " | ".join(d.ljust(10) for d in distances)
		print("-" * len(header))
		print(header)
		print("-" * len(header))
		
		# Print results for each descriptor
		for desc, desc_results in sorted(results.items()):
			row: str = desc.ljust(30) + " | "
			row += " | ".join(f"{desc_results.get(d, {}).get(norm, 0):.4f}".ljust(10) for d in distances)
			print(row)
		print("-" * len(header))

