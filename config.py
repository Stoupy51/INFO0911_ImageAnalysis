
## Configuration file for the application
from src.requirements import *
import time
import os

# Constants for start time
START_TIME: float = time.time()
START_TIME_STR: str = time.strftime("%Y-%m-%d %H:%M:%S")

# Folders
ROOT: str = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")	# Root folder (where the config.py file is located)
IMAGE_FOLDER: str = f"{ROOT}/images"										# Folder where the images are stored
OUTPUT_FOLDER: str = f"{ROOT}/output"										# Folder where the output images will be stored
DATABASE_FOLDER: str = f"{ROOT}/database"									# Folder where the database images are stored

# Other constants
JPG_QUALITY: int = 95



