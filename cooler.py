import cv2
import numpy as np
import os

# Directory containing the images
directory = 'input'

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.endswith('.JPG'):
        # Construct the full file path
        filepath = os.path.join(directory, filename)

        # Read the image
        image = cv2.imread(filepath)

        # Check if the image was successfully loaded
        if image is not None:
            # Apply a cooling filter
            # Increase intensity in the blue channel
            # Decrease intensity in the red channel
            cooled = cv2.addWeighted(image, 0.5, np.zeros(image.shape, image.dtype), 0, -50)

            # Construct the output file path
            output_filepath = os.path.join(directory, 'cooled_' + filename)

            # Save the image
            cv2.imwrite(output_filepath, cooled)
