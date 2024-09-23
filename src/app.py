
## Imports
import matplotlib.pyplot as plt
from distances import DISTANCES_CALLS

# Constants
CHOOSE_DISTANCE: dict[str, str] = {k: k.replace('_',' ').title() for k in DISTANCES_CALLS.keys()}
CHOOSE_DESCRIPTORS: dict[str, str] = {"DC": "Couleurs", "DF": "Formes", "DCNN": "CNN", "DT": "Texture"}
CHOOSE_DESCRIPTOR_CNN: dict[str, str] = {"???": "Faut mettre quoi ici ?"}
CHOOSE_FILTER: dict[str, str] = {"none": "Aucun", "contrast": "Contraste", "sharpness": "Netteté", "blur": "Flou"}
CHOOSE_FILTERS_ON: dict[str, str] = {"input_image": "Image d'entrée", "output_images": "Images de sortie"}

# UI de l'application Shiny
from shiny import ui
app_ui: ui.Tag = ui.page_fluid(

	# Titre de l'application (et de la page)
	ui.panel_title("Recherche d'images par descripteurs"),
	
	# Section de chargement d'image
	ui.layout_sidebar(
		ui.sidebar(

			# Chargement d'image
			ui.input_file("image_upload", "Load image(s)", multiple=True, accept=[".png", ".jpg", ".jpeg"]),
			
			# Sélection du type de distance
			ui.input_select(id="distance", label="Type de distance", choices=CHOOSE_DISTANCE),
			
			# Sélection des descripteurs
			ui.input_checkbox_group(id="descriptors", label="Descripteurs", choices=CHOOSE_DESCRIPTORS),
			
			# Sélection du descripteur CNN
			ui.input_select(id="descripteur_cnn", label="Descripteur CNN", choices=CHOOSE_DESCRIPTOR_CNN),
			
			# Nombre de réponses souhaitées
			ui.input_slider(id="nb_reponses", label="Nombre de réponses souhaitées", min=1, max=100, step=1, value=5),
			
			# Choix du traitement à appliquer
			ui.input_radio_buttons(id="filter_choice", label="Appliquer un filtre", choices=CHOOSE_FILTER),
			ui.input_checkbox_group(id="filters_on", label="Appliquer le filtre sur", choices=CHOOSE_FILTERS_ON, selected=["input_image"]),
			
			# Intensité du filtre
			ui.input_slider(id="filter_intensity", label="Intensité du filtre", min=5, max=100, value=50, step=5),
		),
		
		# Section d'affichage
		ui.page_auto(
			ui.output_text(id="selected_descriptors"),
			ui.output_plot(id="input_image_plot"),
			ui.output_plot(id="output_images_plot"),
		),
	)
)

# Logique backend de l'application
import numpy as np
from shiny import App, render, Session
def server_routine(input: Session, output: Session, app_session: Session) -> None:
	""" Logique backend de l'application Shiny\n
	Args:
		input		(Session):	Inputs de l'application (inputs de l'utilisateur)
		output		(Session):	Outputs de l'application (outputs à afficher)
		app_session	(Session):	Session de l'application (variables de session)
	"""
	# Affiche les descripteurs sélectionnés
	@output
	@render.text
	def selected_descriptors():
		selected: list[str] = input.descriptors()
		joined: str = ', '.join([CHOOSE_DESCRIPTORS[desc] for desc in selected]) if selected else "Aucun"
		return f"Descripteurs sélectionnés : {joined}"
	
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

# Création de l'application Shiny
app: App = App(app_ui, server_routine)

