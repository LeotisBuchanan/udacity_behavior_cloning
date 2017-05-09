import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ImageProcessor import ImageProcessor


class DataGenerator:

    def __init__(self):
        image_processor = ImageProcessor()
        self.augment_func = image_processor.process

    def generator(self, samples_df, batch_size=32):
        num_samples = len(samples_df)
        while 1:  # Loop forever so the generator never terminates
            X_train = []
            y_train = []
            samples_df = samples_df.sample(frac=1).reset_index(drop=True)
            for offset in range(0, num_samples, batch_size):
                batch_samples_df = samples_df[offset:offset + batch_size]
                augmented_df = batch_samples_df.apply(self.getAugmentedSample,
                                                      axis=1)
                # convert df column to array
                images_df = augmented_df[[0]]
                angles_df = augmented_df[[1]]
                X_train = np.vstack(images_df.values)
                y_train = np.vstack(angles_df.values)
                # (8036, 66, 200, 3) (8036,)
                print(X_train.shape, y_train.shape)
                yield X_train[0], y_train.shape

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
    df = pd.read_csv(path)
    images, angles = next(ld.generator(df))
    img = images[0]
    print(img.shape)
    # plt.imshow(images[0])
    # plt.show()
    # print(type(np.array(df["steering"].values)))
    # g = ld.generator(df)
