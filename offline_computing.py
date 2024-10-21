
# Imports
from config import *
from src.print import *
from src.search_engine import offline_cache_compute
import os

# Main function
def main():

	# Print the start time
	info(f"Start time: {START_TIME_STR}")

	# Ask if we shutdown the computer after the script
	shutdown: bool = input("Shutdown the computer after the script? (y/N): ") == "y"

	# Launch the offline computing
	offline_cache_compute()

	# Shutdown the computer
	if shutdown:
		info("Shutdown the computer")
		if os.name == 'nt':
			os.system("shutdown /s /t 1")
		else:
			os.system("sudo shutdown -h now")

	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

