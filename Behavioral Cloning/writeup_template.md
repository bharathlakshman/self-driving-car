# **Behavioral Cloning** 

This project aims at applying deep learning principles to effectively teach a car to drive autonomously in a simulated driving application. The simulator includes both training and autonomous modes. Training mode is used to collect data and  model is created and trained using that data and afterwards, model is tested in autonomous mode. 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)
[image1]: ./images/centre_image.png
[image2]: ./images/left_image.png
[image3]: ./images/right_image.png
[image4]: ./images/center_flipped.png

---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model starts with newtork similar to  known self -driving nVidia model as shown below
![alt text][image1]

My model starts with image shape (160,320,3) instead of (200,66,3) as shown in nVidia architecture.Then, I reproduced this model as shown in image -including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers .The moded also used RELU Activation functions to introduce nonlinearity after each convolution layer .

The model used an adam optimizer, so the learning rate was not tuned manually .
The model was trained and validated on different data sets to ensure that the model was not overfitting 
The results on training and validation data sets are shown below corresponding to epoch value.
![alt text][image2]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , and flipping images to generalize the model better. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to fit the training data on model and testing it on validation data. Initially, I used very simple two layer neural network and fit centre images on that data and run it through simulator and saw the car not able to cross the bridge . I realized that, I needed to provide more information to car.So, I used left and right camera images and adjusted steering angles for these images. I also flipped images and augmented them in the data set to generalize the model better. I also used a cropping layer in my network to crop some portion of image from top and bottom to remove objects such as car hood, trees etc. 

As my data set increased, I decided to use a powerful network similar to Nvidia car model. I evaluated loss on both training set as well as validation set to make sure  mean squared error on the training set and the validation set are close to each other . This implied that the model was not overiftting, 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers
Cropping Layer 
Normalizing layer (Lambda)
Convolution 2D (Kernel_size = (5,5) ,depth = 24 ,strides=(2,2), subsample =(2,2), Activation = relu)
Convolution 2D (Kernel_size = (5,5) ,depth = 36 ,strides=(2,2), subsample =(2,2), Activation = relu)
Convolution 2D (Kernel_size = (5,5) ,depth = 48 ,strides=(2,2), subsample =(2,2), Activation = relu)
Convolution 2D(Kernel_size = (3,3) ,depth = 64 ,strides=(1,1), subsample =(2,2), Activation = relu)
Convolution 2D(Kernel_size = (3,3) ,depth = 64 ,strides=(1,1), subsample =(2,2), Activation = relu)
Fully connected
Dense(100)
Dense(50)
Dense(10)
Dense (1)


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

During this lap, I also recordered images from left and right camera. I used this images with ajdusted steering to augment data. Here is an example of left and right camera images
![alt text][image2]
![alt text][image3]

To augment the data set, I also flipped images and angles thinking that this would help with left turn bais. For example, here is a centre image that has then been flipped:

![alt text][image4]



After the collection process, I had 25712 number of data points. I then preprocessed this data by using normalization.I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 5 .I used an adam optimizer so that manually training the learning rate wasn't necessary.
