
# Imports
from config import *
from src.print import *
from src.test.distances_test import test_all as test_distances

# Main function
def main():

	# Print the start time
	info(f"Start time: {START_TIME_STR} ({START_TIME})")

	# Run all tests
	test_distances()


	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

