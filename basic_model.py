# -*- coding: utf-8 -*-
"""
Basic U-net structure
"""
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
K=tf.keras.backend

#Custom metric
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
X = X[0:y.shape[0],:,:]

#Pad to make a nice shape for convolutions
X_pad=np.zeros([X.shape[0],512,512])
y_pad=np.zeros([X.shape[0],512,512])
X_pad[:, 3:509, :]=X[:,:,12:524]
y_pad[:, 3:509, :]=y[:,:,12:524]

#reshape and one hot encode the labels
X_pad=X_pad.reshape(*X_pad.shape, 1)
y_pad=to_categorical(y_pad, 2)

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

#Classifier
outputs = Conv2D(2, (1,1), activation='softmax')(conv5)

model = Model(inputs=[inputs], outputs=[outputs])

#Compile model with optimiser, loss function etc.
model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy', precision, recall, dice])

model.fit(x=X_train, y=y_train, batch_size=8,epochs=5, validation_data=(X_test, y_test))