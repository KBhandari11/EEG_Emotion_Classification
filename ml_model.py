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
    def __init__(self, input, kern, label, epoch_test, label_test, test_data, test_label):
        self.size = input[0].shape
        label = to_categorical(label,2,dtype='float32')
        label_test = to_categorical(label_test,2, dtype='float32')
        self.test_label = test_label
        self.train_dataset = tf.data.Dataset.from_tensor_slices((input,label))
        self.crosstest = tf.data.Dataset.from_tensor_slices((epoch_test, label_test))
        self.test = test_data
        self.kern = kern        
    

    def train(self):
        BATCH_SIZE = 50
        SHUFFLE_BUFFER_SIZE = 1
        self.train_dataset = self.train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        #self.train_dataset = self.train_dataset.batch(BATCH_SIZE)
        self.crosstest = self.crosstest.batch(BATCH_SIZE)

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
        model.add(layers.Dropout(0.50))
        model.add(layers.Dense(2, activation='softmax'))

        #compiler and train the model
        sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov= True)
        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[tf.keras.metrics.AUC(), 'accuracy'])

        #history = model.fit(self.train_dataset, validation_data= self.crosstest, verbose =1, epochs=100, callbacks=[callback])
        history = model.fit(self.train_dataset, validation_data= self.crosstest, verbose =1, epochs=100)
        test_new = model.predict_classes(self.test)
    
        c = 0
        for x in range(self.test_label.shape[0]):
            if(test_new[x] == self.test_label[x]):
                c+= 1
        model_json = model.to_json()
        with open("./modeloutput/valence_frequency6.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("./modeloutput/valence_frequency6.h5")
        print("Saved model to disk")

        #return (history, c/self.test_label.shape[0])
        return (history, 0)
        
    