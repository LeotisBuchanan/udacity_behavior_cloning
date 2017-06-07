import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from ImageProcessor import ImageProcessor
from enum import Enum

"""
def my_exit(msg):
    import sys
    print(msg)
    sys.exit(0)
"""

class DataGenerator:


    def __init__(self):

        self.LEFT, self.CENTER, self.RIGHT = 0, 1, 2
        self.SMALL_SCALE = 0.9
        self.LARGE_SCALE = 1.1

        image_processor = ImageProcessor()
        self.augment_func = image_processor.process

    def generator(self, samples, batch_size=32):

        num_samples = len(samples)

        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    image, angle = self.getAugmentedSample(batch_sample)
                    images.append(image)
                    angles.append(float(angle))

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

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
        image = plt.imread(path.strip())
        # new_angle = float(steering_angle) + steering_correction


        image, steering_angle = self.augment_func(image, l_angle, self.LEFT)
        image, steering_angle = self.augment_func(image, r_angle, self.RIGHT)

        return image, steering_angle



    def getAugmentedSample(self, row):

        #['IMG/center_2016_12_01_13_36_20_507.jpg', ' IMG/left_2016_12_01_13_36_20_507.jpg',
        # ' IMG/right_2016_12_01_13_36_20_507.jpg', ' 0.04262284', ' 0.9855326', ' 0', ' 30.18659']


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
    path = "combine_data/driving_log_2.csv"
    ld = DataGenerator()

    samples = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_gen = ld.generator(train_samples)
    valid_gen = ld.generator(validation_samples)
    x_train, y_train = next(train_gen)
    img = x_train[0]
    ang  = y_train[0]

    print(ang)
    print(img)



