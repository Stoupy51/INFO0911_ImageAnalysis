
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
IMAGE_EXTENSIONS: tuple[str] = (".jpg", ".jpeg", ".png")					# Image extensions to consider
OUTPUT_EXTENSION: str = ".jpg"												# Output image extension
JPG_QUALITY: int = 95

#TODO: Reduce image size for the search engine
SEARCH_MAX_IMAGE_SIZE: tuple[int, int] = (448, 448)

