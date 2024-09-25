
# Imports
import os
import sys

# Try to import every requirements
REQUIREMENTS: list[str] = ["numpy", "pandas", "pillow", "shiny", "seaborn", "faicons", "pydicom"]
EXIT_PROGRAM: bool = False
for requirement in REQUIREMENTS:
	try:
		__import__(requirement) if requirement != "pillow" else __import__("PIL")
	except ImportError:
		os.system(f"{sys.executable} -m pip install {requirement}")
		EXIT_PROGRAM = True

# Exit program if any requirement has been installed
if EXIT_PROGRAM:
	print("Please restart the program.")
	sys.exit(-1)

