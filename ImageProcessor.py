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

    def crop_image(self, image, y1, y2, x1, x2):
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image

    def process(self, image, steering_angle):
        """
         Apply processing to image
        """
        # image size
        im_x = image.shape[1]

        # translate image and compensate for steering angle
        trans_range = 50
        image, steering_angle = self.random_translate(image,
                                                      steering_angle,
                                                      trans_range)

        # crop unwanted parts of image
        image = self.crop_image(image, 20, 140,
                                trans_range, im_x - trans_range)

        # resize the image after cropping
        image = cv2.resize(image, (200, 66))

        # flip image (randomly)
        if np.random.uniform() >= 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle

        # augment brightness
        image = self.augment_brightness(image)

        return image, steering_angle


if __name__ == "__main__":
    image = mpimg.imread("data/IMG/left_2016_12_01_13_39_24_588.jpg")
    steering_angle = 0
    ip = ImageProcessor()
    image, sa = ip.process(image, steering_angle)
    print(image.shape)
    plt.imshow(image)
    plt.show()
