
## Imports
from src.processing.extensive_tests import extensive_tests
from src.print import *

# Constants
NB_ITERATIONS: list[int] =	[15, 20, 50]				# Nombre d'itérations
KAPPA: list[int] = 			[15, 50, 100]				# Seuil de contraste
GAMMA: list[float] =		[0.1, 0.25, 0.5, 1.0]		# Coefficient de conduction
FORMULA: list[int] =		[1,2]						# 1 utilise exp, 2 utilise la fonction inverse
NOISES: list[str] = [
	"speckle_0_200",	# moyenne 0, écart-type 200
	"speckle_0_80",		# moyenne 0, écart-type 80
	"speckle_0_50",		# moyenne 0, écart-type 50

	"none"				# Pas de bruit
]

if __name__ == "__main__":
	info("Début des tests extensifs")
	extensive_tests(NOISES, NB_ITERATIONS, KAPPA, GAMMA, FORMULA)
	info("Fin des tests extensifs")

