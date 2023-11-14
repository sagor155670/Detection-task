import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths

# Read the image
# image = cv2.imread("inputs/images (6).jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imshow("rgb",image)
# print(image.shape)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
# for (name, channel) in zip(("L","A","B"), cv2.split(image)):
#     cv2.imshow(name,channel)

# print(image.shape)
# cv2.imshow("rgb",image)
input_folder = "input"
image_paths = list(paths.list_images(input_folder))
diff = []
outlier = []
outlier2 = []
test = []
for path in image_paths:
    # Load the image
    image = cv2.imread(path)

    # Split the image into its color channels
    b,g,r = cv2.split(image)

    red = np.array(r)
    green = np.array(g)
    blue = np.array(b)

    std_R = np.std(red) 
    std_G = np.std(green)
    std_B = np.std(blue)

    # print(f"{path.split('/')[-1]} -> red: {std_R} blue: {std_B}")
    print(f"{path.split('/')[-1]} -> diff - {abs(std_B - std_R)}")
    diff.append(abs(std_B - std_R))
    if abs(std_B - std_R) > 15:
        outlier.append(path.split('/')[-1])
    if abs(std_B - std_R) >= 20 and abs(std_B - std_R) <=25:
        outlier2.append(path.split('/')[-1])
    if  abs(std_B - std_R) >= 0 and abs(std_B - std_R) <= 1:  
        test.append(path.split('/')[-1]) 
# print('mean', np.array(diff).mean())    
print(len(outlier))
# print(f'outlier2 len: {len(outlier2)}')
print(f'length: {len(diff)}')
# print(outlier2)
# print(test)
# print(len(test))


# channels = cv2.split(image)
# colors = ("b", "g", "r")

# plt.figure()
# plt.title("Color Balance Check")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")

# # Loop over each channel
# for (channel, color) in zip(channels, colors):
#     # Create a histogram for the current channel and plot it
#     histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
#     plt.plot(histogram, color=color)
#     plt.xlim([0, 256])

#     # Find the peak of the histogram and print it
#     peak = np.argmax(histogram)
#     print(f'Peak for {color} channel: {peak}')

# def on_key(event):
#     if event.key == 'escape':
#         plt.close(event.canvas.figure)

# # Connect the function to the key press event
# cid = plt.gcf().canvas.mpl_connect('key_press_event', on_key)



# # Plot the histogram
# plt.show()

# # Disconnect the function from the key press event
# plt.gcf().canvas.mpl_disconnect(cid)

# cv2.destroyAllWindows()

