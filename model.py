from keras.layers.core import (Dense, Flatten, Dropout, Lambda)
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import json
from keras.models import Sequential
from datagenerator import DataGenerator
import csv


class ModelManager:

    def __init__(self):
        dg = DataGenerator()
        self.generator = dg.generator


    def defineModel(self, image_shape):
        # i think the problem is with this image size
        ch, row, col = 160, 320, 3
        model = Sequential()
        # Preprocess incoming data, centered around zero with small standard
        # deviation
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=(ch, row, col),
                         output_shape=(ch, row, col)))

        model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

        model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
        model.add(ELU())


        model.add(Convolution2D(48, 3, 3, subsample=(2, 2)))
        model.add(ELU())


        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(ELU())
        model.add(Dropout(0.2))

        model.add(Convolution2D(128, 3, 3, subsample=(1, 1)))
        model.add(ELU())
        
        model.add(Flatten())

        model.add(Dense(100))
        model.add(ELU())
                
        model.add(Dense(50))
        model.add(ELU())
        

        model.add(Dense(10))
        model.add(ELU())

        model.add(Dense(1))

        opt = Adam(lr=0.00007)
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
        model.summary()

        return model

    def trainModel(self, samples_df, model_path="model"):
        # compile and train the model using the generator function

        print("training model")
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
                            nb_epoch=100)

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
