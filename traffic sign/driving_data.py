# Load pickled data
import pickle
# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(np.append(np.append(y_train,y_test,axis=0),y_valid,axis=0)))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import numpy as np
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
plt.figure(figsize=(2,2))
plt.imshow(image)
print('Image label is : ',y_train[index])


def count_sign(y):
    labels=np.zeros(n_classes)
    for i in range(0,len(y)):
        labels[y[i]] +=1
    return labels
    
y_train_dis= count_sign(y_train)
y_test_dis= count_sign(y_test)
y_valid_dis= count_sign(y_valid)
print(y_train_dis)
'''
import matplotlib.pyplot as plt
print('Train set labels\' distribution:')
plt.bar(range(len(y_train_dis)), y_train_dis)
plt.show()
print('Test set labels\' distribution:')
plt.bar(range(len(y_test_dis)), y_test_dis)
plt.show()
print('Valid set labels\' distribution:')
plt.bar(range(len(y_valid_dis)), y_valid_dis)
plt.show()  
'''
#max_images_per_label=np.max(y_train_dis)
max_images_per_label=2500
import cv2
import random

##Helper functions
'''
Data augmentation techniques I have used:
random rotate - -10 ~ +10 degree
random scale - 0.8 ~ 1.2 scale
random translate - -2 ~ +2 pixel
random shear - -5 ~ +5 degree
random brightness - 0.5 ~ 2.0 factor
'''
def image_rotate(img, angle):
    """Rotate the image by angle"""
    rows, cols, dims = img.shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, matrix, (cols, rows))


def image_scale(img, scale):
    """Adjust the image size by scale"""
    rows, cols, dims = img.shape
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
    return cv2.warpAffine(image, matrix, (cols, rows))


def image_translate(img, x, y):
    """Translate image by the value of x and y"""
    rows, cols, dims = img.shape
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, matrix, (cols, rows))


def image_shear(img, shear_range):
    """Shear image randomly by the factor of shear_range"""
    rows, cols, dims = img.shape
    
    pts1 = np.float32([[5, 5],[20, 5],[5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    matrix = cv2.getAffineTransform(pts1, pts2)
    
    return cv2.warpAffine(img, matrix, (cols, rows))


def image_brightness(image, factor):
    """Adjust the brightness of images by factor"""
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = np.minimum(hsv[:,:,2] * factor, 255)
    
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)


def random_image_transform(image):
    """Transform image according to given parameters"""
    randomAngle = random.uniform(-10, 10)
    output = image_rotate(image, randomAngle)
    
    randomScale = random.uniform(0.8, 1.2)
    output = image_scale(output, randomScale)
    
    randomFactor = random.uniform(0.5, 2.0)
    output = image_brightness(output, randomFactor)

    randomX = random.uniform(-2, 2)
    randomY = random.uniform(-2, 2)
    output = image_translate(output, randomX, randomY)
    
    randomShear = random.uniform(-5, 5)
    output = image_shear(output, randomShear)
    
    return output


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


#Generate fake data
new_train=[]
new_label=[]

while np.sum(y_train_dis) < n_classes * max_images_per_label:
    for index, image in enumerate(X_train):
        img_class = y_train[index]
        
        if y_train_dis[img_class] < max_images_per_label:
            
            new_train.append(random_image_transform(image))
            new_label.append(img_class)
            y_train_dis[img_class] += 1

X_train = np.append(X_train, new_train, axis=0)
y_train = np.append(y_train, new_label, axis=0)
'''
#Normalize data
import numpy as np

a = (X_train[...,0:1]-128)/128
b = (X_train[...,1:2]-128)/128
c = (X_train[...,2:3]-128)/128
X_train=np.concatenate((a,b,c),axis=3)


a = (X_valid[...,0:1]-128)/128
b = (X_valid[...,1:2]-128)/128
c = (X_valid[...,2:3]-128)/128
X_valid=np.concatenate((a,b,c),axis=3)

a = (X_test[...,0:1]-128)/128
b = (X_test[...,1:2]-128)/128
c = (X_test[...,2:3]-128)/128
X_test=np.concatenate((a,b,c),axis=3)
'''

#RGB to grayscale
from skimage.color import rgb2gray
X_train = np.array([rgb2gray(img).reshape(32,32,1) for img in X_train])
X_valid = np.array([rgb2gray(img).reshape(32,32,1) for img in X_valid])
X_test = np.array([rgb2gray(img).reshape(32,32,1) for img in X_test])


#plt.imshow(X_train[0], cmap="gray")
#plt.show()  

import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 256
KEEP_PROB= 0.5

### Define your architecture here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten


def TSCNet(x,keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    #Activation.
    conv1 = tf.nn.relu(conv1)

    #Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    #Activation.
    conv2 = tf.nn.relu(conv2)
	#droput
    conv2    = tf.nn.dropout(conv2, keep_prob)
    #Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    #Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    #Activation.
    fc1    = tf.nn.relu(fc1)
    #dropout
    fc1    = tf.nn.dropout(fc1, keep_prob)
    
    #Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    #Activation.
    fc2    = tf.nn.relu(fc2)
    #dropout
    fc2    = tf.nn.dropout(fc2, keep_prob)
    
    #Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    l2_loss = tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv1_b) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(conv2_b)+tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc1_b)+tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc2_b)+tf.nn.l2_loss(fc3_W) + tf.nn.l2_loss(fc3_b)
    return logits,l2_loss

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

keep_prob = tf.placeholder(tf.float32)
rate = 0.002

logits,l2loss = TSCNet(x,keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
#Regularation  be care of this parameter,may prove 
beta = 0.002
loss_operation = tf.reduce_mean(cross_entropy)+beta*l2loss
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        #loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})
        #print("Evaluate Loss = {:.3f}".format(loss))
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:KEEP_PROB})
            #loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:1.0})
            #print("Train Loss = {:.3f}".format(loss))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './tscnet')
    print("Model saved")
