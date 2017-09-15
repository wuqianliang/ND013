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
from sklearn.utils import shuffle
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
from sklearn.model_selection import train_test_split

# Work on GTX 1060 with memory 6G
'''
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
'''

# input image with 320 pixel width 160 pixel height and three color channels
ch = 3
row = 160
col = 320

BATCHSIZE=16

# use generator function to save menmory when prepare train or valid batchs of image
def generator(samples, impage_path='./IMG/' ,batch_size=BATCHSIZE, step_finetune = False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
				# center camera
                name = impage_path+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if step_finetune:
                    correction = 0.1 # this is a parameter to tune

                    # left camera
                    name = impage_path+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    left_angle = float(batch_sample[3]) + float(correction)
                    images.append(left_image)
                    angles.append(left_angle)

                    # right camera
                    name = impage_path+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)
                    right_angle = float(batch_sample[3]) - float(correction)
                    images.append(right_image)
                    angles.append(right_angle)
                

			# data augmentation to prevent data bias for the reason that in first track, car always tern left 
            augmented_images,augmented_measurements = [],[]
            for image,measurement in zip(images,angles):

				# resize image down by scale facter 0.5
                #image = _image # cv2.resize(_image,(col,row))
                augmented_images.append(image)
                augmented_measurements.append(measurement)
				# flip images and take the opposite sign of the steering measurement to help with the left turn bias
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)


            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_nvidia_model_1():

	model = Sequential()
	# normalize the image data
	model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, ch),output_shape=(row, col, ch)))

	# crop each image to focus on only the portion of the image that contains road section
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Activation('relu'))
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	print('Model is created and compiled..')
	return model

if __name__ == "__main__":

	########################step 1 ################################
	#
	#  train new model
	#
	###############################################################
	
	#read data logs from csv file
	samples = []
	with open('./driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)

	# split dataset to train and valid parts
	train_samples, validation_samples = train_test_split(samples, test_size=0.3)

	# the model use the generator function to make more memory-efficient
	#train data generator
	train_generator = generator(train_samples,impage_path='./IMG/',batch_size=BATCHSIZE)

	#valid data generater
	validation_generator = generator(validation_samples,impage_path='./IMG/',batch_size=BATCHSIZE)

	_model= create_nvidia_model_1()
	print(_model.summary())

	print(len(train_samples),BATCHSIZE)

	history_object = _model.fit_generator(
		train_generator, 
		steps_per_epoch=len(train_samples)/BATCHSIZE,
		epochs=4,
		verbose=1,
		validation_data = validation_generator,
		validation_steps=len(validation_samples)/BATCHSIZE,
		max_queue_size=10,
		samples_per_epoch = len(train_samples)
		)

	_model.save('./model.h5')
	
	########################step 2 ################################
	#
	#  finetuning model for some spots where there is sharp turn 
	#
	###############################################################

    # check that model Keras version is same as local Keras version
	f = h5py.File('./model.h5', mode='r')
	model_version = f.attrs.get('keras_version')
	keras_version = str(keras_version).encode('utf8')
	f.close()
	if model_version != keras_version:
		print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

	_model = load_model('./model.h5')
	_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss="mse")
	samples = []
	with open('./IMG_SPOT1/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)

	# split dataset to train and valid parts
	train_samples, validation_samples = train_test_split(samples, test_size=0.3)

	# the model use the generator function to make more memory-efficient
	#train data generator
	train_generator = generator(train_samples,impage_path='./IMG_SPOT1/IMG/',batch_size=BATCHSIZE, step_finetune = True)

	#valid data generater
	validation_generator = generator(validation_samples,impage_path='./IMG_SPOT1/IMG/',batch_size=BATCHSIZE, step_finetune = True)

	print(len(train_samples),BATCHSIZE)

	history_object = _model.fit_generator(
		train_generator, 
		steps_per_epoch=len(train_samples)/BATCHSIZE,
		epochs=8,
		verbose=1,
		validation_data = validation_generator,
		validation_steps=len(validation_samples)/BATCHSIZE,
		max_queue_size=10,
		samples_per_epoch = len(train_samples)
		)

	_model.save('./model.h5')
