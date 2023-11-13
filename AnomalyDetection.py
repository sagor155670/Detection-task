#Necessary imports
from imutils import paths
import cv2
import colour
import numpy as np
from scipy.signal import convolve2d
import math

class AnomalyDetection:
    def __init__(self):
        pass
    def variance_of_laplacian(self,image):
        blur_map = cv2.Laplacian(image,cv2.CV_64F)
        variance = blur_map.var()
        return blur_map, variance
    
    def isBlurred(self, image_path, threshold):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map ,laplacian_variance = self.variance_of_laplacian(gray)
        save_path = "output/" + image_path.split('/')[-1]
        print(save_path)
        cv2.imwrite(save_path,blur_map)
        print("Blur amount: ", laplacian_variance)
        if laplacian_variance < threshold:
            return True
        return False
    
    def estimate_noise(self, image):
        H,W = image.shape
        M = [
            [1,-2,1],
            [-2,4,-2],
            [1,-2,1]
        ]
        sigma = np.sum(np.sum(np.absolute(convolve2d(image,M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
        print("Noise amount -> sigma: ",sigma)
        return sigma

    def isNoisy(self, image_path, threshold):
         image = cv2.imread(image_path)
         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
         noise = self.estimate_noise(gray)
         return noise > threshold
    
    def isGreyish(self, image):
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_image)
        # print('mean: ',np.array(s).mean())
        std_s = np.array(s).std()
        # print('std(s): ',s)
        if np.all(s <  5) or std_s < 20:
            return std_s, True
        else:
            return std_s,False

    
    def isColorBalanced(self,image_path, threshold = 15):
        image = cv2.imread(image_path)
        
        # Split the image into its color channels
        b,g,r = cv2.split(image)
        # converting the channels to numpy array
        red = np.array(r)
        blue = np.array(b)
        # calculating standard deviation
        std_R = np.std(red) 
        std_B = np.std(blue)   
        # checking if image is greyish or not
        std, isGrey = self.isGreyish(image=image)
        diff = abs(std_B - std_R)  
        print('diff of R & B:', diff)
        # if diff is grater than threshold then image is not color balanced
        if diff > threshold or isGrey:
            return std, False
        return std, True
       
  



    
if __name__ == '__main__':
    detect = AnomalyDetection()
    input_folder = "input"
    image_paths = list(paths.list_images(input_folder))
    # img = Image.open('inputs/images (5).jpg')
    # img_conv = detect.convert_to_srgb(img)
    # image = cv2.imread("inputs/warm1.jpeg")
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    std, res = detect.isColorBalanced(image_path="warms/warmed_50.jpeg" )
    print('std: ',std)
    print(res)
    # warmStd = []
    # for path in image_paths:
    #     std, result = detect.isColorBalanced(path)
    #     warmStd.append(std)
    #     print(result)
    # print(np.array(warmStd).mean())
    # print(np.array(warmStd).std())
    
    # print(np.array(warmStd))

    

    # for image in image_paths:
    #     is_blurred = detect.isBlurred(image_path=image, threshold=100)
    #     print(image, "Blurred: ", is_blurred)
    # for image in image_paths:
    #     is_noisy = detect.isNoisy(image_path=image, threshold=8.0)
    #     print("image name:",image.split('/')[-1], "Noisy:",is_noisy)
    



