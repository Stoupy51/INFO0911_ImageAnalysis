
# Import config from the parent folder
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
import pandas as pd

# Imports
from src.print import *
from src.mean_average_precision import evaluate_descriptors, print_evaluation_results
from src.color_space.all import COLOR_SPACES_CALLS
from src.descriptors import DESCRIPTORS_CALLS
from src.distances import DISTANCES_CALLS

def main() -> None:
	""" Main function """
	# Print the start time
	info(f"Start time: {START_TIME_STR}")

	# Get available options
	color_spaces: list[str] = list(COLOR_SPACES_CALLS.keys())
	descriptors: list[str] = list(DESCRIPTORS_CALLS.keys())
	distances: list[str] = list(DISTANCES_CALLS.keys())

	info("Available options:")
	info(f"Color spaces: {color_spaces}")
	info(f"Descriptors: {descriptors}")
	info(f"Distances: {distances}")
	print()

	# Load existing results
	results_dir: str = "evaluation_results"
	results_file: str = f"{results_dir}/results.csv"
	if os.path.exists(results_file):
		results_df = pd.read_csv(results_file)
		info(f"Loaded {len(results_df)} existing results from {results_file}")
	
	# Ask for evaluation parameters
	plot_curves: bool = input("Generate precision-recall curves? (y/N): ").lower() == 'y'
	use_k: bool = input("Use MAP@K? (y/N): ").lower() == 'y'
	k: int|None = int(input("Enter K value: ")) if use_k else None

	# Ask if we want to evaluate all combinations
	evaluate_all: bool = input("Evaluate all combinations? (y/N): ").lower() == 'y'

	if evaluate_all:
		# Evaluate all combinations
		info("Evaluating all combinations...")
		results = evaluate_descriptors(
			color_spaces=color_spaces,
			descriptors=descriptors,
			distances=distances,
			k=k,
			plot_curves=plot_curves
		)
	else:
		# Ask for specific combinations
		print("\nSelect color spaces (comma-separated):")
		selected_cs: list[str] = input(f"Options {color_spaces}: ").strip().split(',')
		selected_cs = [cs.strip() for cs in selected_cs if cs.strip() in color_spaces]
		if not selected_cs:
			selected_cs = [list(COLOR_SPACES_CALLS.keys())[0]]

		print("\nSelect descriptors (comma-separated):")
		selected_desc: list[str] = input(f"Options {descriptors}: ").strip().split(',')
		selected_desc = [d.strip() for d in selected_desc if d.strip() in descriptors]
		if not selected_desc:
			selected_desc = [list(DESCRIPTORS_CALLS.keys())[0]]

		print("\nSelect distances (comma-separated):")
		selected_dist: list[str] = input(f"Options {distances}: ").strip().split(',')
		selected_dist = [d.strip() for d in selected_dist if d.strip() in distances]
		if not selected_dist:
			selected_dist = [list(DISTANCES_CALLS.keys())[0]]

		# Validate selections
		if not (selected_cs and selected_desc and selected_dist):
			error("Invalid selection. Please select at least one option for each category.")
			return

		# Run evaluation
		info("Starting evaluation...")
		results = evaluate_descriptors(
			color_spaces=selected_cs,
			descriptors=selected_desc,
			distances=selected_dist,
			k=k,
			plot_curves=plot_curves
		)

	# Print results
	print_evaluation_results(results)

	if plot_curves:
		info("Precision-recall curves saved to precision_recall_curves.png")

	# End of the script
	info("End of the script")

# Entry point of the script
if __name__ == "__main__":
	main()

