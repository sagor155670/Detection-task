import cv2
import numpy as np
from imutils import paths
import os

input_folder = "input"
image_paths = list(paths.list_images(input_folder))
output = 'equalized/'
for path in image_paths:
    # Load the image
    image = cv2.imread(path)
 
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Merge the channels back together
    image_eq = cv2.merge([b_eq, g_eq, r_eq])

    output_filepath = os.path.join(output, 'eq_' + path.split('/')[-1])
    # Save the image
    cv2.imwrite(output_filepath, image_eq)
