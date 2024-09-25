## Imports
from config import *
import numpy as np
from PIL import Image
from src.processing.cedf import cedf_filter
from src.processing.fastaniso import anisodiff
import cv2

images: list[str] = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg",".png"))]
niter = 15
k = 100
g = 0.1
fonction = 2

# For each image,
for image in images:
    base_image = f"{IMAGE_FOLDER}/{image}"
    speckle_0_50 = f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_50/_speckle_0_50.jpg"
    speckle_0_80 = f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_80/_speckle_0_80.jpg"
    speckle_0_200 = f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_200/_speckle_0_200.jpg"

    #image sans bruit
    base_image_enhanced = cedf_filter(base_image)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/none/_enhanced.jpg", base_image_enhanced.astype(np.uint8))
    print(base_image,"done")

    #image avec bruit 50
    speckle_0_50_enhanced = cedf_filter(speckle_0_50)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_50/_enhanced.jpg", speckle_0_50_enhanced.astype(np.uint8))
    print(speckle_0_50,"done")
    #image avec bruit 50, après fastaniso
    noised_image = np.array(Image.open(speckle_0_50))
    if noised_image.ndim == 3:
        noised_image = np.mean(noised_image, axis=-1)
    output: np.ndarray = anisodiff(noised_image, niter=niter, kappa=k, gamma=g, option=fonction, ploton=False)
    speckle_0_50_PM_enhanced = cedf_filter(output)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_50/_pm_enhanced.jpg", speckle_0_50_PM_enhanced.astype(np.uint8))
    print(speckle_0_50,"PM done")

    #image avec bruit 80
    speckle_0_80_enhanced = cedf_filter(speckle_0_80)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_80/_enhanced.jpg", speckle_0_80_enhanced.astype(np.uint8))
    print(speckle_0_80,"done")
    #image avec bruit 80, après fastaniso
    noised_image = np.array(Image.open(speckle_0_80))
    if noised_image.ndim == 3:
        noised_image = np.mean(noised_image, axis=-1)
    output: np.ndarray = anisodiff(noised_image, niter=niter, kappa=k, gamma=g, option=fonction, ploton=False)
    speckle_0_80_PM_enhanced = cedf_filter(output)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_80/_pm_enhanced.jpg", speckle_0_80_PM_enhanced.astype(np.uint8))
    print(speckle_0_80,"PM done")

    #image avec bruit 200
    speckle_0_200_enhanced = cedf_filter(speckle_0_200)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_200/_enhanced.jpg", speckle_0_200_enhanced.astype(np.uint8))
    print(speckle_0_200,"done")
    #image avec bruit 200, après fastaniso
    noised_image = np.array(Image.open(speckle_0_200))
    if noised_image.ndim == 3:
        noised_image = np.mean(noised_image, axis=-1)
    output: np.ndarray = anisodiff(noised_image, niter==niter, kappa=k, gamma=g, option=fonction, ploton=False)
    speckle_0_200_PM_enhanced = cedf_filter(output)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{image[:-4]}/speckle_0_200/_pm_enhanced.jpg", speckle_0_200_PM_enhanced.astype(np.uint8))
    print(speckle_0_200,"PM done")

