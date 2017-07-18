import time
import csv
import cv2
import sys
import numpy as np
import matplotlib.image as mpimg
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import pickle
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from sklearn.model_selection import train_test_split
from dataGenerator import * 

class ModelGenerator(object):

    def __init__(self):
        self.IMAGE_HEIGHT= 160
        self.IMAGE_CHANNELS = 3
        self.IMAGE_WIDTH = 320

    def buildModel(self):
        model = Sequential()
        model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((50,20),(0,0)))) # (top,bottom),(left,right)
        model.add(Conv2D(24,(5,5),strides = (2,2), activation = "relu"))
        model.add(Conv2D(36,(5,5),strides = (2,2), activation = "relu"))
        model.add(Conv2D(48,(5,5),activation = "relu"))
        model.add(Conv2D(64,(3,3),activation = "relu"))
        model.add(Conv2D(64,(3,3),activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.summary()
        model.compile(loss="mse", optimizer = "adam", metrics=["accuracy"])
        return model

    def trainModel(self, model,samples, output_path,epochs=10):

        (train_samples,validation_samples) = train_test_split(samples,test_size=0.2)
        dgen = DataGenerator(self.IMAGE_HEIGHT,
                             self.IMAGE_WIDTH,
                             self.IMAGE_CHANNELS)
        
        validation_generator = dgen.validationGenerator(validation_samples)
        train_generator  = dgen.trainGenerator(train_samples)
    

        ts = str(int(time.time()))
        output_path = output_path + "model_"+ ts + ".h5"
        print("writing model to:" + output_path)
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50)
        # history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040)
        model.save(output_path)  # save model to .h5 file, including architechture, weights, loss, optimizer
        


def main(args):
    if len(args) < 2:
        print("please enter input and output path")
        print("python model.py path/to/data  path/to/output")
        sys.exit(0)
    driving_log_file = args[0]
    driving_log_path = args[1]
    samples = []
    output_path= args[2]
    with open(driving_log_file) as csv_file:
        for line in csv_file:
            line = line.split(",")
            samples.append(line)
            
    # split the samples into valid and test
    
    model_gen = ModelGenerator()
    model = model_gen.buildModel()
    model_gen.trainModel(model,samples,output_path)


    

if __name__ == "__main__":
    main(sys.argv[1:])

