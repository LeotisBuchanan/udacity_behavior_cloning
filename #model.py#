from keras.layers.core import (Dense, Flatten, Dropout, Lambda)
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import Cropping2D
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import json
from keras.models import Sequential
from datagenerator import DataGenerator
from keras import layers
from keras import models
import csv


class ModelManager:

    def __init__(self):
        dg = DataGenerator()
        self.generator = dg.generator

    def defineModel2(self, INPUT_SHAPE):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1,activation='relu'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['accuracy'])
        return model


        
    def defineModel(self, INPUT_SHAPE):
         model = Sequential()
         model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
         model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
         model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
         model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
         model.add(Conv2D(64, 3, 3, activation='elu'))
         model.add(Conv2D(64, 3, 3, activation='elu'))
         model.add(Conv2D(64, 3, 3, activation='elu'))
         model.add(Conv2D(128, 3, 3, activation='elu'))
         model.add(Dropout(0.1))
         model.add(Flatten())
         model.add(Dense(100, activation='elu'))
         model.add(Dense(50, activation='elu'))
         model.add(Dense(10, activation='elu'))
         model.add(Dense(1))
         model.summary()
         model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['accuracy'])
         return model


    def trainModel(self, samples_df, model_path="model"):
        # compile and train the model using the generator function

        print("training model")
        (train_samples_df,
         validation_samples_df) = train_test_split(samples_df,
                                                   test_size=0.2)

        train_generator = self.generator(train_samples_df)
        validation_generator = self.generator(validation_samples_df)

        image_input_shape = (160, 320, 3)
        # image_input_shape = 66, 200, 3
        model = self.defineModel2(image_input_shape)
        # train model
        checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
        """
        model.fit_generator(train_generator,
                            callbacks=[checkpoint],
                            samples_per_epoch=len(train_samples_df),
                            validation_data=validation_generator,
                            nb_val_samples=len(validation_samples_df),
                            nb_epoch=3)
        """
        model.fit_generator(train_generator,
                            callbacks=[checkpoint],
                            samples_per_epoch=20000,
                            validation_data=validation_generator,
                            nb_val_samples=2000,
                            nb_epoch=10)



        # save the model
        model.save_weights(model_path + '_weights.h5', True)
        model.save(model_path + '.h5')
        with open(model_path + '.json', 'w') as outfile:
            json.dump(model.to_json(), outfile)


    
if __name__ == "__main__":
    import sys
    samples = []
    data_path = "data/driving_log_100.csv"
    data_path = "data/driving_log_path_fixed.csv"
    #data_path = "data/driving_log.csv"
    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(line)
    mm = ModelManager()
    tg = mm.trainModel(samples)
