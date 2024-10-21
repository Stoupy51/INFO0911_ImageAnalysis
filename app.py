
# Imports
from config import *
from src.print import *
import os

# Main function
def main():

	# Print the start time
	info(f"Start time: {START_TIME_STR}")

	# Start the shiny app
	SHINY_APP_PATH: str = f"{ROOT}/src/app.py"
	os.system(f"shiny run --launch-browser {SHINY_APP_PATH}")

	# End of the script
	info("End of the script")
	return



# Entry point of the script
if __name__ == "__main__":
	main()

