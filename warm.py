import cv2
import numpy as np
import os


# Directory containing the images
directory = 'input/'

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is an image
   if filename.endswith('.JPG') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.webp') or filename.endswith('.avif') or filename.endswith('.bmp'):

        # Construct the full file path
        filepath = os.path.join(directory, filename)

        # Read the image
        image = cv2.imread(filepath)

        # Check if the image was successfully loaded
        if image is not None:
            # Split the image into its color channels
            b, g, r = cv2.split(image)

            # Apply a warming filter
            # Increase intensity in the red and green channels
            r = cv2.addWeighted(r, 1.5, np.zeros(r.shape, r.dtype), 0, 0)
            g = cv2.addWeighted(g, 1.3, np.zeros(g.shape, g.dtype), 0, 0)

            # Decrease intensity in the blue channel
            b = cv2.addWeighted(b, 0.8, np.zeros(b.shape, b.dtype), 0, 0)

            # Merge the channels back together
            warmed = cv2.merge((b, g, r))

            # Construct the output file path
            output_filepath = os.path.join(directory, 'warmed_' + filename)

            # Save the image
            cv2.imwrite(output_filepath, warmed)

