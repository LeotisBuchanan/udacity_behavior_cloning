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

class ModelGenerator(object):

    def __init__(self):
        pass

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

    def trainModel(self, model,X_train, y_train, output_path,epochs=10):
        ts = str(int(time.time()))
        output_path = output_path + "model_"+ ts + ".h5"
        print("writing model to:" + output_path)
        model.fit(X_train, y_train, validation_split=0.2,shuffle = True, epochs=epochs)
        model.save(output_path)  # save model to .h5 file, including architechture, weights, loss, optimizer
        

    def saveToPickle(self,pickle_path,X_train, y_train):
        with open(pickle_path,'wb') as f:
            pickle.dump((X_train,y_train), f, pickle.HIGHEST_PROTOCOL)


    def readTrainingDataFromPickle(self,data_pickle_path):
        X_train, y_train = None, None
        try:
            with open(data_pickle_path, 'rb') as f:
                X_train, y_train
                data = pickle.load(f)
                X_train = data[0]
                y_train  = data[1]
            return X_train, y_train
        except FileNotFoundError as e:
            print(e)
            sys.exit(0)


    
    def readFile(self,driving_log_path,driving_log_file,pickle_path, read_from_pickle=True):
        X_train, y_train = None, None
        images = []
        measurements = []
        if read_from_pickle:
            #read the data from pickle
            X_train, y_train  = self.readTrainingDataFromPickle(pickle_path)
        else:
            with open(driving_log_path+"/" + driving_log_file) as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader) # skip header['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
                for line in reader:
                    source_path = line[0]
                    filename = source_path.split("/")[-1]
                    current_path = driving_log_path+ "/IMG/" + filename
                    rgb = mpimg. imread(current_path)
                    images.append(rgb)
                    measurement = float(line[3])  # steer
                    measurements.append(measurement)

                X_train = np.array(images)  #(8036, 160, 320, 3)
                y_train = np.array(measurements) #(8036,)
                self.saveToPickle(pickle_path,X_train, y_train)
        return X_train, y_train

def main(args):
    if len(args) < 2:
        print("please enter input and output path")
        print("python model.py path/to/data  path/to/output")
        sys.exit(0)
    print(args)
    driving_log_file = args[0]
    driving_log_path = args[1]
    output_path= args[2]
    model_gen = ModelGenerator()
    pickle_path = driving_log_path + "data.pickle"
    X_train, y_train = model_gen.readFile(driving_log_path,driving_log_file,pickle_path)
    print(X_train.shape)
    print(y_train.shape)
    model = model_gen.buildModel()
    model_gen.trainModel(model,X_train,y_train,output_path)


    

if __name__ == "__main__":
    main(sys.argv[1:])

