
# Imports
from config import *
from src.print import *
from src.search_engine import offline_cache_compute
import os

# Main function
def main():

	# Print the start time
	info(f"Start time: {START_TIME_STR}")

	# Launch the offline computing
	offline_cache_compute()

	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

