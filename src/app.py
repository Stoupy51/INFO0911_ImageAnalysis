
## Imports
from search_engine import search, COLOR_SPACES_CALLS, DESCRIPTORS_CALLS, DISTANCES_CALLS
from shiny import ui

# Constants
CHOOSE_COLOR_SPACE: dict[str, str] = {k: k.replace('_',' ').title() for k in COLOR_SPACES_CALLS.keys()}
CHOOSE_DISTANCE: dict[str, str] = {k: k.replace('_',' ').title() for k in DISTANCES_CALLS.keys()}
CHOOSE_DESCRIPTORS: dict[str, str] = {k: k.replace('_',' ').title() for k in DESCRIPTORS_CALLS.keys()}

# UI de l'application Shiny
app_ui: ui.Tag = ui.page_fluid(

	# Titre de l'application (et de la page)
	ui.panel_title("Recherche d'images"),
	
	# Section de chargement d'image
	ui.layout_sidebar(
		ui.sidebar(

			# Color space selection
			ui.input_select(id="color_space", label="Type d'espace de couleur", choices=CHOOSE_COLOR_SPACE),

			# Descriptors selection
			ui.input_select(id="descriptor", label="Type de descripteur", choices=CHOOSE_DESCRIPTORS),
			
			# Distance selection
			ui.input_select(id="distance", label="Type de distance", choices=CHOOSE_DISTANCE),
			
			# Nombre de réponses souhaitées
			ui.input_numeric(id="nb_reponses", label="Nombre de réponses souhaitées", min=1, max=100, step=1, value=10),

			# Image upload
			ui.input_file("image_upload", "Load image(s)", multiple=True, accept=[".png", ".jpg", ".jpeg"]),
		),
		
		# Section d'affichage
		ui.page_auto(
			ui.output_plot(id="input_image_plot"),
			ui.output_plot(id="output_images_plot"),
		),

		# Sidebar always open
		always_open=True
	)
)

# Remaining imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shiny import App, render, Session

# Logique backend de l'application
def server_routine(input: Session, output: Session, app_session: Session) -> None:
	""" Logique backend de l'application Shiny\n
	Args:
		input		(Session):	Inputs de l'application (inputs de l'utilisateur)
		output		(Session):	Outputs de l'application (outputs à afficher)
		app_session	(Session):	Session de l'application (variables de session)
	"""	
	# Affiche l'image d'entrée
	@output
	@render.plot
	def input_image_plot():
		images: list[dict] = input.image_upload()	# [{'name': 'image.jpg', 'size': 134068, 'type': 'image/jpeg', 'datapath': 'C:/...'}]
		if not images:
			return None

		# Prepare the plot
		fig, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(5 * len(images), 5))
		fig: plt.Figure
		axs: np.ndarray

		# Set the title
		fig.suptitle("Image d'entrée" if len(images) == 1 else "Images d'entrée")
		
		# For each image, display it
		for i, image in enumerate(images):
			datapath: str = image['datapath']

			# Display the image
			subplot: plt.Axes = axs[i] if len(images) > 1 else axs
			subplot.imshow(plt.imread(datapath))
			subplot.axis('off')
			subplot.set_title(image['name'])

		# Return the plot
		return fig
	
	# Affiche les images de sortie
	@output
	@render.plot
	def output_images_plot():
		# Get the inputs for the search engine
		images: list[dict] = input.image_upload()
		if not images:
			return None
		image: Image.Image = Image.open(images[0]['datapath']).convert("RGB")
		image_request: np.ndarray = np.array(image)
		color_space: str = input.color_space()
		descriptor: str = input.descriptor()
		distance: str = input.distance()
		max_results: int = input.nb_reponses()

		# Search for similar images
		results: list[tuple[str, np.ndarray, float]] = search(image_request, color_space, descriptor, distance, max_results)

		# Prepare the plot
		MAX_PER_ROW: int = 5
		nrows: int = (len(results) + MAX_PER_ROW - 1) // MAX_PER_ROW
		figsize: tuple[int, int] = (5 * MAX_PER_ROW, 5 * nrows)
		fig, axs = plt.subplots(nrows=nrows, ncols=MAX_PER_ROW, figsize=figsize)
		fig: plt.Figure
		axs: np.ndarray

		# Set the title
		fig.suptitle("Images similaires")

		# For each result, display it
		for i, (path, image, distance) in enumerate(results):
			# Display the image
			row: int = i // MAX_PER_ROW
			col: int = i % MAX_PER_ROW
			subplot: plt.Axes = axs[row, col] if nrows > 1 else axs[col]
			subplot.imshow(image)
			subplot.axis('off')
			image_name: str = path.split('/')[-1]
			subplot.set_title(f"{image_name} ({distance:.2f})")
		
		# Return the plot
		return fig


# Création de l'application Shiny
app: App = App(app_ui, server_routine)

