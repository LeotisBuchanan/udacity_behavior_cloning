import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from ImageProcessor import ImageProcessor


class DataGenerator:
    def __init__(self):

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
                    
                    name = 'data/IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def getAugmentedImage(self, camera_name, row,
                          augment_func, steering_correction):
        path_prefix = "data/"
        path = row["left"]
        steering_angle = row["steering"]
        path = path_prefix + path.strip()
        image = plt.imread(path.strip())
        new_angle = float(steering_angle) + steering_correction
        image, steering_angle = self.augment_func(image, new_angle)

        return image, steering_angle

    def getAugmentedSample(self, row):

        cam_view = np.random.choice(['center', 'left', 'right'])

        image = None
        steering_angle = None
        if cam_view == 'left':
            # left image
            image, steering_angle = self.getAugmentedImage("left",
                                                           row,
                                                           self.augment_func,
                                                           0.25)

        elif cam_view == 'center':
            # centre image
            image, steering_angle = self.getAugmentedImage("center",
                                                           row,
                                                           self.augment_func,
                                                           0)

        elif cam_view == 'right':
            # right image
            image, steering_angle = self.getAugmentedImage("right",
                                                           row,
                                                           self.augment_func,
                                                           -0.25)

        return pd.Series([image, steering_angle])


if __name__ == "__main__":
    path = "data/driving_log.csv"
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
    print(img.shape)
    print(ang)
    plt.imshow(img)
    plt.show()

