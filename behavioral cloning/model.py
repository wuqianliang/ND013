from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Cropping2D
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import csv
import cv2
import os


ch = 3
row = 160
col = 320
BATCHSIZE=64
#width 320 height 160 channel 3


samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=BATCHSIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

			#data augmentation
            augmented_images,augmented_measurements = [],[]
            for image,measurement in zip(images,angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)


            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)




# the model use the generator function
train_generator = generator(train_samples, batch_size=BATCHSIZE)
validation_generator = generator(validation_samples, batch_size=BATCHSIZE)



def create_nvidia_model_1():

	model = Sequential()
	#data preprocess
	model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, ch),output_shape=(row, col, ch)))
	#cropping
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Activation('relu'))
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(50))
	model.add(Activation('relu'))
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	print('Model is created and compiled..')
	return model

if __name__ == "__main__":


	_model= create_nvidia_model_1()
#	history_object = _model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=3)	
	history_object = _model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=3, verbose=1)
	_model.save('./model.h5')
	### print the keys contained in the history object
#'''
	print(history_object.history.keys())
	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
#'''