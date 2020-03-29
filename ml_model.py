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
    def __init__(self, input, kern, label, epoch_train, label_train):
        self.size = input[0].shape
        print(self.size)
        label = to_categorical(label,2,dtype='float32')
        label_train = to_categorical(label_train,2, dtype='float32')
        self.train_dataset = tf.data.Dataset.from_tensor_slices((input,label))
        self.test = tf.data.Dataset.from_tensor_slices((epoch_train, label_train))
        self.kern = kern        
    

    def train(self):
        BATCH_SIZE = 20
        SHUFFLE_BUFFER_SIZE = 1
        self.train_dataset = self.train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test = self.test.batch(BATCH_SIZE)

        model = models.Sequential()
        #first layer
        model.add(layers.Conv2D(64, self.kern, activation='relu',padding='valid', input_shape=(self.size)))
        model.add(layers.MaxPooling2D((2, 2)))
        #second layer
        model.add(layers.Conv2D(64, self.kern,padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #summary of the model
        model.summary()
        #Flatten or also full connection layer
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax',kernel_initializer='random_uniform',
                bias_initializer='ones'))


        #compiler and train the model
        sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)
        #model.compile(optimizer=sgd, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, name='binary_crossentropy'), metrics=['acc'])
        model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
        valence_history = model.fit(self.train_dataset,  epochs=20)
        model.evaluate(self.test, verbose=2)