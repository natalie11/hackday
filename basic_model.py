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

#Load data and labels
data_filename = "C:/Users/Natal/Documents/CABI/ML/Vessel data/fadus_subvol/fadus_deconv_subvol.npy"
label_filename  = "C:/Users/Natal/Documents/CABI/ML/Vessel data/fadus_subvol/fadus_deconv_subvol_labels.npy"
print('Loading labels from '+str(data_filename))
X = np.load(data_filename)
y = np.load(label_filename)
X = X[0:y.shape[0],:,:]

#Split into training and test images
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Define Model
inputs = Input((X_train.shape[1],X_train.shape[2], 1))

#Encoding branch
conv1 = Conv2D(32, (3,3), activation='relu')(inputs)
conv1 = Conv2D(32, (3,3), activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
drop1 = Dropout(0.25)(pool1)

conv2 = Conv2D(64, (3,3),activation='relu')(drop1)
conv2 = Conv2D(64, (3,3), activation='relu')(conv2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
drop2 = Dropout(0.25)(pool2)

conv3=Conv2D(128, (3,3), activation='relu')(drop2)
conv3=Conv2D(64, (3,3), activation='relu')(conv3)

up4 = Conv2DTranspose(64, (2,2), strides=(2,2))(conv3)
concat4 = concatenate([up4, conv2], axis=3)
conv4 = Conv2D(64, (3,3), activation='relu')(concat4)
drop2 = Dropout(0.25)(conv4)

up5 = Conv2DTranspose(32, (2,2), strides=(2,2))(conv4)
concat5 = concatenate([up5, conv1], axis=3)
conv5 = Conv2D(32, (3,3), activation='relu')(concat5)

outputs = Conv2D(2, (1,1,1), activation='softmax')(conv5)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metric='accuracy')

model.fit(X=X_train, y=y_train, batch_size=8,epochs=5)