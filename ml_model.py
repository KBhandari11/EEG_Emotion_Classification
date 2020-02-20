from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class cnn_model:
    def __init__(self, input, filter, valence, arousal, validation):
        self.input = input
        self.filter = filter        
        self.valence = valence
        self.arousal = arousal
        self.validation = validation

    def train():
        model = models.Sequential()
        #first layer
        model.add(layers.Conv2D(128, filter, activation='relu', data_,input_shape=(19,2400,128)))
        model.add(layers.MaxPooling2D((2, 2)))
        #second layer
        model.add(layers.Conv2D(64, filter, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #summary of the model
        model.summary()
        #Flatten or also full connection layer
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        #compiler and train the model
        model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['AUC'])
        valence_history = model.fit(self.input, self.valence, epochs=10, validation_data=(test_images, test_labels))
        arousal_history = model.fit(self.input, self.arousal, epochs=10, validation_data=(test_images, test_labels))

        #Evaluation of the Model 
        #valence 
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['alidation_AUC'], label = 'validation_AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)