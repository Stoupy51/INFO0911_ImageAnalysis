
# Imports
import os
import sys

# Try to import every requirements
REQUIREMENTS: list[str] = ["numpy","shiny"]
for requirement in REQUIREMENTS:
	try:
		__import__(requirement)
	except ImportError:
		os.system(f"{sys.executable} -m pip install {requirement}")
		print("Please restart the program.")
		sys.exit(-1)

