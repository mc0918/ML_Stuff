# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:39:31 2019

@author: mcamp
"""
##requires tensorflow package installed on system

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print("hello world")

# dataset with grayscale images (28 x 28 pixels) divided into 10 categories
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#10 class names that will be represented by integers 0-9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#60,000 images in training set, with each represented as 28 x 28 pixels
print(train_images.shape)

#60,000 labels in training set
print(len(train_labels))

#each label is an integer between 0 and 9
print(train_labels)

#10,000 images in testing set, each 28 x 28 pixels
print(test_images.shape)

#testing set contains 10,000 image labels
print(len(test_labels))

#The data must be preprocessed before training the network
#Below is a plot showing pixel values in a range of 0 to 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Divide these values (0-255) by 255 to get a range of 0 to 1 for preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

#Verify the data is in the correct format
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#train the neural network
#The first layer flattens the data from 28 x 28 pixels from a 2D array to a 1D array of 784 pixels
#The next 2 layers are densely-connected neural layers.
#2nd layer has 128 nodes (neurons)
#3rd layer has 10 nodes that return probability scores that sum to 1
#These represent the probability that the image belongs to 1 of the 10 classes
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(10,activation = tf.nn.softmax)  
])
    
#Compile the model to minimize loss of accuracy
#metrics = accuracy means the model uses the number of images correctly identified to quantify success
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']) 
 
#start training by running the model 5 times
model.fit(train_images, train_labels, epochs=5)

#compare how the model performs on the test dataset now that it's been trained
#model will likely do worse on test due to overfitting
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#Now, with the model trained, we can use it to make predictions about specific images
predictions = model.predict(test_images)
predictions[0]
print("-----------------------------")

#Find the first image
print(np.argmax(predictions[0]))

#then check test_label to see if the prediction is correct
print(test_labels[0])
print("-----------------------------")

#Graph to look at the full set of all 10 channels
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
              100*np.max(predictions_array),
              class_names[true_label]),
              color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
#With the graph defined, let's look at the 0th image
#high confidence is represented by the blue bar
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions,test_labels)
plt.show()

#On the other hand, now let's look at an image incorrectly identified
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions,test_labels)
plt.show()

#Plotting several images and their predictions at once
#correct predictions in blue, incorrect in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

#Use the trained model to make predictions about a single image
img = test_images[0]
print(img.shape)

#keras makes predictions about a collection of examples, so a single image must be added to a list
img = (np.expand_dims(img,0))
print(img.shape)

#Predict the image
predictions_single = model.predict(img)
print(predictions_single)

#Plot confidence in prediction's classification
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#Grab the actual class of the image
print(np.argmax(predictions_single[0]))
#the model predicts image 0 is an ankle boot (class_name 9)...
#and the model is right!

