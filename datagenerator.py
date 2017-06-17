import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from ImageProcessor2 import ImageProcessor
from enum import Enum
from globals import *

def my_exit(msg):
    import sys
    print(msg)
    sys.exit(0)

class DataGenerator:


    def __init__(self):
        self.LEFT, self.CENTER, self.RIGHT = 0, 1, 2
        self.SMALL_SCALE = 0.9
        self.LARGE_SCALE = 1.1
        self.INPUT_SHAPE = IMAGE_SHAPE
        self.IMAGE_HEIGHT = self.INPUT_SHAPE[0]
        self.IMAGE_WIDTH = self.INPUT_SHAPE[1]
        self.IMAGE_CHANNELS = self.INPUT_SHAPE[2]
        print("input shape :" + str(self.INPUT_SHAPE)) 


        image_processor = ImageProcessor()
        self.augment_func = image_processor.augment

    def generator(self, samples, batch_size=32):


        num_samples = len(samples)
        images = np.empty([batch_size, self.IMAGE_HEIGHT,
                           self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steer_angles = np.empty(batch_size)
        while 1:  # Loop forever so the generator never terminates
            samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                i = 0 
                for batch_sample in batch_samples:
                    image, angle = self.getAugmentedSample(batch_sample)
                    images[i] = image
                    steer_angles[i] = angle
                    i = i + 1
                    if i ==  batch_size:
                        break
                yield images, steer_angles

    def getAugmentedImage(self, camera_name, row,
                          augment_func, steering_correction):

        path_prefix = "data/"
        image_path = row[camera_name]
        image  = None
        steering_angle = None


        STEERING_IDX = 3
        angle = float(row[STEERING_IDX])

        if angle > 0:  # right turn
            l_angle = self.LARGE_SCALE * angle
            r_angle = self.SMALL_SCALE * angle
        else:  # left turn
            l_angle = self.SMALL_SCALE * angle
            r_angle = self.LARGE_SCALE * angle

        path = path_prefix + image_path.strip()
        image = mpimg.imread(path.strip())
        


        image, steering_angle = self.augment_func(image, l_angle, self.LEFT)
        image, steering_angle = self.augment_func(image, r_angle, self.RIGHT)

        return image, steering_angle



    def getAugmentedSample(self, row):

        cam_view = np.random.choice([self.LEFT, self.CENTER, self.RIGHT])


        image = None
        steering_angle = None
        if cam_view == self.LEFT:

            # left image
            image, steering_angle = self.getAugmentedImage(self.LEFT,
                                                           row,
                                                           self.augment_func,
                                                           0.25)

        elif cam_view == self.CENTER:
            # centre image
            image, steering_angle = self.getAugmentedImage(self.CENTER,
                                                           row,
                                                           self.augment_func,
                                                           0)

        elif cam_view == self.RIGHT:
            # right image
            image, steering_angle = self.getAugmentedImage(self.RIGHT,
                                                           row,
                                                           self.augment_func,
                                                           -0.25)
        return image, steering_angle


if __name__ == "__main__":
    import sys
    path = "data/driving_log_path_fixed.csv"
    # path = "data/driving_log.csv"
    ld = DataGenerator()
    samples = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        count = 0 
        for line in reader:
            samples.append(line)
            count = count + 1
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_gen = ld.generator(train_samples)
    valid_gen = ld.generator(validation_samples)
    count = 0
    for x, y in train_gen:
        print("imges :" + str(len(x)))
        print("angles: " + str(len(y)))
        img = x[0]
        ang = y[0]
        plt.imshow(img)
        plt.show()
        count = count + 1
        if count > 10:
            break
    my_exit("exiy")





