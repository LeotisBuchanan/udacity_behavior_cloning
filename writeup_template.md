---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py models/track1/model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the following model architecture: 


conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
conv2d_3 (Conv2D)            (None, 16, 73, 48)        43248
conv2d_4 (Conv2D)            (None, 14, 71, 64)        27712
conv2d_5 (Conv2D)            (None, 12, 69, 64)        36928
dropout_1 (Dropout)          (None, 12, 69, 64)        0
flatten_1 (Flatten)          (None, 52992)             0
dense_1 (Dense)              (None, 100)               5299300
dense_2 (Dense)              (None, 50)                5050
dense_3 (Dense)              (None, 10)                510
dense_4 (Dense)              (None, 1)                 11

The model is implemented in model.py file in the buildModel method. 
As seen in the implementation relu layers to introduce nonlinearity after all convolutional layers. 



####2. Attempts to reduce overfitting in the model

My model contains a single dropout layer, having a value of 0.5. This layer was introduced when I noticed that the train model was performing better on the training data than on the validation data. This is often an indication that the model is overfitting. Thus I added a dropout layer to reduce this overfitting.


#### Augmenting the data: 
I did not used the provided udacity data set. Instead I used the simulator to collect my data. My approach to collecting this data was perform alot of sharp left and right turns. I believe doing emphasized the importance of not running off the road. 

I first tried to train the model with a lot of augmented data. All of these attempts were unsuccessfull. Eventually in dispair I tried the training the model with small amount of data, collected as described above. I also only use the center image(because who really drives with 3 cameras). 

It worked. You can see the video of the result here : <insert video>

####3. Model parameter tuning

My approach to parameter tuning was as follows:
- Implement an existing architecture.
-  observer it performance on the track
- based on the performance I would adjust model parameters(dropout %, convolutional filter size etc)

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

I spent hours gathering trainig data, recovering from left, recovering from right etc. Augmenting the data etc. Then I decided to just try using only the center camera. I also decided to drive in such away that my training data contain a lot of recovering from left and right.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
