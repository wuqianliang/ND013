import keras
import pandas as pd
import numpy as np
import csv
import cv2
import os,math
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Cropping2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.utils import plot_model
from sklearn.utils import shuffle



# input image with 320 pixel width 160 pixel height and three color channels
ch = 3
row = 160
col = 320
BATCHSIZE=64


#read data logs from csv file
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split dataset to train and valid parts
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)




# use generator function to save menmory when prepare train or valid batchs of image
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

			#data augmentation to prevent data bias for the reason that in first track, car always tern left 
            augmented_images,augmented_measurements = [],[]
            for image,measurement in zip(images,angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
				# flip images and take the opposite sign of the steering measurement to help with the left turn bias
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)


            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)




# the model use the generator function to make more memory-efficient

#train data generator
train_generator = generator(train_samples, batch_size=BATCHSIZE)

#valid data generater
validation_generator = generator(validation_samples, batch_size=BATCHSIZE)

def create_nvidia_model_1():

	model = Sequential()
	#normalize the image data
	model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, ch),output_shape=(row, col, ch)))

	# crop each image to focus on only the portion of the image that contains road section
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
	model.add(Activation('relu'))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
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

def create_nvidia_model_2():

	model = Sequential()

	#data preprocess
	model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, ch),output_shape=(row, col, ch)))

	#cropping
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", 
		kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
		kernel_regularizer=regularizers.l2(0.001),
		activity_regularizer=regularizers.l1(0.001),
		input_shape=(row, col, ch)))

	model.add(Activation('relu'))

	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same",
		kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
		kernel_regularizer=regularizers.l2(0.001),
		activity_regularizer=regularizers.l1(0.001)))

	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same",
		kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
		kernel_regularizer=regularizers.l2(0.001),
		activity_regularizer=regularizers.l1(0.001)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same",
		kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
		kernel_regularizer=regularizers.l2(0.001),
		activity_regularizer=regularizers.l1(0.001)))

	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, 3, 3, 
		subsample=(2, 2), border_mode="same",
		kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
		kernel_regularizer=regularizers.l2(0.001),
		activity_regularizer=regularizers.l1(0.001)))
	model.add(Flatten())
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(
		Dense(100,
		kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
		kernel_regularizer=regularizers.l2(0.001),
		activity_regularizer=regularizers.l1(0.001)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(50,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)))
	model.add(Activation('relu'))
	model.add(Dense(1,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)))

	model.compile(optimizer="adam", loss="mse")
#	model.compile(optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004), loss="mse")

	print('Model is created and compiled..')
	return model

'''
def step_decay(epoch):
    initial_lrate = 0.002
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
sgd = SGD(lr=0.002, momentum=0.9, decay=0.0, nesterov=False)
'''

if __name__ == "__main__":

	_model= create_nvidia_model_2()
	print(_model.summary())

	print(len(train_samples),BATCHSIZE)

	history_object = _model.fit_generator(
		train_generator, 
		steps_per_epoch=len(train_samples)/BATCHSIZE,
		epochs=3,
		verbose=1,
		validation_data = validation_generator,
		validation_steps=len(validation_samples)/BATCHSIZE,
		max_queue_size=10,
		samples_per_epoch = len(train_samples)
		)

	_model.save('./model.h5')

	with open('log_adam_64.txt','w') as f:
		f.write(str(history.history))

'''
	### print the keys contained in the history object

	history_object = _model.fit_generator(
		train_generator, 
		steps_per_epoch=len(train_samples)/BATCHSIZE,
		epochs=3,
		verbose=1,
		callbacks=[lrate],
		validation_data = validation_generator,
		validation_steps=len(validation_samples)/BATCHSIZE,
		max_queue_size=10,
		samples_per_epoch = len(train_samples)
		)


	print(history_object.history.keys())
	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
'''