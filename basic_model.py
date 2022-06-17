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
data_filename = "path to data file"
label_filename  = "path to label file"
X = np.load(data_filename)
Y=np.load(label_filename)

#Split into training and test images
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

#Define Model
inputs = Input((X_train.shape[1],X_train.shape[2], 1))

#Encoding branch
conv1 = Conv2D(32, (3,3), activation='relu')(inputs)