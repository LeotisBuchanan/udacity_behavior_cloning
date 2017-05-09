from keras.layers.core import (Dense, Flatten, Dropout)
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import json
from keras.models import Sequential
from datagenerator import DataGenerator
import pandas as pd


class ModelManager:

    def __init__(self):
        dg = DataGenerator()
        self.generator = dg.generator

    def defineModel(self, image_shape):
        ch, row, col = image_shape
        model = Sequential()
        # Preprocess incoming data, centered around zero with small standard
        # deviation
        # model.add(Lambda(lambda x: x / 127.5 - 1.,
        #                 input_shape=(ch, row, col),
        #                 output_shape=(ch, row, col)))

        model.add(Convolution2D(36, 5, 5, subsample=(2, 2),
                                input_shape=(66, 200, 3)))

        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Convolution2D(64, (3, 3), subsample=(2, 2)))
        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Convolution2D(48, (3, 3), subsample=(2, 2)))
        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Convolution2D(64, (3, 3), subsample=(1, 1)))
        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Convolution2D(64, (3, 3), subsample=(1, 1)))
        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(100))
        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Dense(50))
        model.add(ELU())
        model.add(Dropout(0.5))

        model.add(Dense(10))
        model.add(ELU())

        model.add(Dense(1))

        opt = Adam(lr=0.00007)
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
        model.summary()

        return model

    def trainModel(self, samples_df, model_path="model"):
        # compile and train the model using the generator function

        (train_samples_df,
         validation_samples_df) = train_test_split(samples_df,
                                                   test_size=0.2)

        train_generator = self.generator(train_samples_df)
        validation_generator = self.generator(validation_samples_df)

        image_input_shape = 66, 200, 3
        model = self.defineModel(image_input_shape)
        # train model
        model.fit_generator(train_generator,
                            samples_per_epoch=len(train_samples_df),
                            validation_data=validation_generator,
                            nb_val_samples=len(validation_samples_df),
                            nb_epoch=3)

        # save the model
        model.save_weights(model_path + '_weights.h5', True)
        model.save(model_path + '.h5')
        with open(model_path + '.json', 'w') as outfile:
            json.dump(model.to_json(), outfile)


if __name__ == "__main__":
    data_path = "data/driving_log.csv"
    samples_df = pd.read_csv(data_path)
    mm = ModelManager()
    tg = mm.trainModel(samples_df)
