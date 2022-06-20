# -*- coding: utf-8 -*-
"""
Basic U-net structure
"""
#Import all neccesary libraries 
import os
import numpy as np
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
K=tf.keras.backend

#Define custom metrics, to be recorded during training
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def dice(y_true, y_pred):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    dice = 2*(P*R)/(P+R+K.epsilon())
    return dice


#Load data and labels
data_filename = "C:/Users/Natal/Documents/CABI/ML/Vessel data/fadus_subvol/fadus_deconv_subvol.npy"
label_filename  = "C:/Users/Natal/Documents/CABI/ML/Vessel data/fadus_subvol/fadus_deconv_subvol_labels.npy"
print('Loading labels from '+str(data_filename))
X = np.load(data_filename)
y = np.load(label_filename)

#Oops, I didn't have labels for the last few slices of my image!
#This bit of code is just cropping the data to have the same number of images as the labels
X = X[0:y.shape[0],:,:]

#Pad to make a nice shape for convolutions
X_pad=np.zeros([X.shape[0],512,512])
y_pad=np.zeros([X.shape[0],512,512])
#I worked this out manually for my image size- I'm padding the x-axis with 3 zeros on each side, and cropping the y-axis by 12 on each side
X_pad[:, 3:509, :]=X[:,:,12:524]
#Same for the labels
y_pad[:, 3:509, :]=y[:,:,12:524]

#reshape and one hot encode the labels
X_pad=X_pad.reshape(*X_pad.shape, 1) #add an extra dimension where the labels will end up in the prediction
y_pad=to_categorical(y_pad, 2)

#Double check that the shapes make sense - should be (no. of images, 512, 512, 1) and (no. of images, 512, 512, 2)
print("Shape of X:" +str(X_pad.shape))
print("Shape of y:" +str(y_pad.shape))

#Split into training and test images
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size = 0.25)

#Define Model
inputs = Input((X_train.shape[1],X_train.shape[2], 1))

#Encoding branch
conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
drop1 = Dropout(0.25)(pool1)

conv2 = Conv2D(64, (3,3),activation='relu', padding='same')(drop1)
conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
drop2 = Dropout(0.25)(pool2)

#Bottom of the 'U'
conv3=Conv2D(128, (3,3), activation='relu', padding='same')(drop2)
conv3=Conv2D(64, (3,3), activation='relu', padding='same')(conv3)

#Decoding branch
up4 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv3)
concat4 = concatenate([up4, conv2], axis=3)
conv4 = Conv2D(64, (3,3), activation='relu', padding='same')(concat4)
drop4 = Dropout(0.25)(conv4)

up5 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv4)
concat5 = concatenate([up5, conv1], axis=3)
conv5 = Conv2D(32, (3,3), activation='relu', padding='same')(concat5)

#Classifier (2 filters for 2 classes)
outputs = Conv2D(2, (1,1), activation='softmax')(conv5)

model = Model(inputs=[inputs], outputs=[outputs])

#Compile model with chosen optimiser, loss function etc.
model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy', precision, recall, dice])

#Train for a set number of epochs
model.fit(x=X_train, y=y_train, batch_size=8, epochs=1)

#Evaluate the model on the test data
model.evaluate(X_test,y_test, batch_size=8)

#Run a prediction
predicted_labels = model.predict(X_test)

#Reverse the one hot encoding (by finding the index of the max value in the last dimension)
predicted_labels = np.argmax(predicted_labels, axis=3)

#Save the output images
path = "C:/Users/Natal/Documents/CABI/ML/Vessel data/fadus_subvol/"
prediction_filename = "fadus_prediction"
#Cycle through the image stack, saving each image as a .tif (you don't have to do it this way, it's just what I find easiest for viewing the results)
for im in range (predicted_labels.shape[0]):
  filename = os.path.join(path,str(im+1)+"_"+str(prediction_filename)+".tif")
  image = Image.fromarray(predicted_labels[im,:,:], mode='1')
  image.save(filename)