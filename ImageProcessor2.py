import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from  globals import *

class ImageProcessor:

    def __init__(self, w=320, h=160):
        self.IMAGE_HEIGHT = IMAGE_SHAPE[0]
        self.IMAGE_WIDTH = IMAGE_SHAPE[1]
        


    def crop(self,image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom)
        """
        return image[60:-25, :, :] # remove the sky and the car front


    def resize(self,image):
        """
        Resize the image to the input shape used by the network model
        """
        return cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), cv2.INTER_AREA)


    def rgb2yuv(self,image):
        """
        Convert the image from RGB to YUV (This is what the NVIDIA model does)
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


    def preprocess(self,image):
        """
        Combine all preprocess functions into one
        """
        print("***** before crop" + str(image.shape))
        image = self.crop(image)
        print("***** after crop" + str(image.shape))

        #image = self.resize(image)
        # image = self.rgb2yuv(image)
        return image


    def choose_image(data_dir, center, left, right, steering_angle):
        """
        Randomly choose an image from the center, left or right, and adjust
        the steering angle.
        """
        steering_angle = float(steering_angle)
        choice = np.random.choice(3)
        if choice == 0:
            return load_image(data_dir, left), steering_angle + 0.2
        elif choice == 1:
            return load_image(data_dir, right), steering_angle - 0.2
        return load_image(data_dir, center), steering_angle


    def random_flip(self,image, steering_angle):
        """
        Randomly flipt the image left <-> right, and adjust the steering angle.
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle


    def random_translate(self,image, steering_angle, range_x=10, range_y=10):
        """
        Randomly shift the image virtially and horizontally (translation).
        """
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle


    

    def random_shadow(self,image):
        """
        Generates and adds random shadow
        """
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = self.IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = self.IMAGE_WIDTH * np.random.rand(), self.IMAGE_HEIGHT
        xm, ym = np.mgrid[0:self.IMAGE_HEIGHT, 0:self.IMAGE_WIDTH]

        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line: 
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


    def random_brightness(self,image):
        """
        Randomly adjust brightness of the image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


    def augment(self,image, steering_angle, range_x=10, range_y=10):
        """
        Generate an augumented image and adjust steering angle.
        (The steering angle is associated with the center image)
        """
        # image, steering_angle = self.random_flip(image, steering_angle)
        # image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)
        #image = self.random_shadow(image)
        #image = self.random_brightness(image)
        return image, steering_angle



if __name__ == "__main__":
    image = mpimg.imread("data/IMG/center_2017_05_24_18_14_12_653.jpg")
    print("source " + str(image.shape))
    steering_angle = 0
    ip = ImageProcessor()
    image =  ip.preprocess(image)
    image, sa = ip.augment(image, steering_angle)
    print("after augment " + str(image.shape))
    plt.imshow(image)
    plt.show()
