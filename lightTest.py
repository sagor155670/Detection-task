import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import os
from PIL import Image,ImageStat

input_folder = "darks"
image_paths = list(paths.list_images(input_folder))
output = 'lights'

def brightness( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.rms[0]

brights = []

for path in image_paths:
    # Load the image
    light = int(brightness(path))
    print(f'{path.split("/")[-1]}: {light}')
    brights.append(light)
    # image = cv2.imread(path)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(hsv)
    # output_filepath = os.path.join(output, 'light_' + path.split('/')[-1])
    # cv2.imwrite(output_filepath,v)


brightsData = np.array(brights)
print('length: ', len(brights) )
indices = np.array(range(len(brightsData)))
plt.boxplot(brightsData)
# Create a histogram
# plt.hist(brightsData, bins=range(min(brightsData), max(brightsData) + 2))
# plt.scatter(indices,brightsData)
# plt.plot(brightsData)
plt.show()
print(np.array(brights).std())