from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,metrics ,optimizers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CNN_Model:
    def __init__(self, input, kern, valence, arousal):
        self.train_dataset_valence = input
        self.valence = to_categorical(valence)
        print(self.valence.shape)
        self.kern = kern        

    def train(self):
        model = models.Sequential()
        #first layer
        model.add(layers.Conv2D(128, self.kern, activation='relu', input_shape=(self.train_dataset_valence[0].shape)))
        model.add(layers.MaxPooling2D((2, 2)))
        #second layer
        model.add(layers.Conv2D(64, self.kern, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #summary of the model
        model.summary()
        #Flatten or also full connection layer
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.15))
        model.add(layers.Dense(2, activation='softmax'))

        #compiler and train the model
        #sgd = optimizers.SGD(lr=0.0001)
        #model.compile(optimizer=sgd, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, name='binary_crossentropy'), metrics=['acc'])
        model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        valence_history = model.fit(self.train_dataset_valence, self.valence,batch_size=20,  epochs=40)

     