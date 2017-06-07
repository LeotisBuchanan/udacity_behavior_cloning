import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class ImageProcessor:

    def __init__(self):
        pass

    def augment_brightness(self, image):
        """
         apply random brightness on the image
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()

        # scaling up or down the V channel of HSV
        image[:, :, 2] = image[:, :, 2] * random_bright
        return image

    def random_translate(self, image, steer, trans_range):
        """
        randomly translate a image
        """

        rows, cols, chan = image.shape

        # horizontal translation with 0.008 steering compensation per pixel
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        steering_angle = steer + tr_x / trans_range * .4

        tr_y = 0

        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

        return image_tr, steering_angle

    def darker_img(self, image):
        # Convert to YUV
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_gray = img_yuv[:, :, 0]

        # Pick the majority pixels of the image
        idx = (img_gray < 245) & (img_gray > 10)

        # Make the image darker
        img_gray_scale = img_gray[idx] * np.random.uniform(0.1, 0.6)
        img_gray[idx] = img_gray_scale

        # Convert back to BGR
        img_yuv[:, :, 0] = img_gray
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

    def process(self, image, steering_angle,CAMERA):

        """
         Apply processing to image
        """
        # image size
        im_x = image.shape[1]

        # flip image (randomly)
        if np.random.uniform() >= 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle

        # randomly darken the image
        if np.random.uniform() >= 0.5:
            image = self.darker_img(image)

        # augment brightness
        #image = self.augment_brightness(image)

        return image, steering_angle


if __name__ == "__main__":
    image = mpimg.imread("data/IMG/left_2016_12_01_13_39_24_588.jpg")
    image = cv2.imread("data/IMG/left_2016_12_01_13_39_24_588.jpg")
    steering_angle = 0
    ip = ImageProcessor()
    image, sa = ip.process(image, steering_angle)
    print(image.shape)
    plt.imshow(image)
    plt.show()
