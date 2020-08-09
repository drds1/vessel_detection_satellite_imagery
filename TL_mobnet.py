# coding: utf-8

#load dependencies
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam




base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model.trainable = False

output_layers = keras.Sequential()
output_layers.add(keras.layers.GlobalAveragePooling2D())
output_layers.add(keras.layers.Dense(32,activation='relu'))
#output_layers.add(keras.layers.Dense(1024,activation='relu'))
output_layers.add(keras.layers.Dense(3,activation='softmax'))

final_model = keras.Sequential()
# add resnet model
final_model.add(base_model)
# add the trainable output layers
final_model.add(output_layers)


final_model.summary()