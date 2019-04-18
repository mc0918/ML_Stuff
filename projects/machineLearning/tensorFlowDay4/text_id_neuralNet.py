# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:52:44 2019

@author: mcamp
"""
##Made in Spyder development environment

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#download imdb revies dataset
imdb = keras.datasets.imdb

#downloads the 10,000 most common words in reviews to keep data size manageable
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

#Each word is assigned an integer, representing the word in a dictionary
#Each label is either 0 (negative review) or 1 (positive review)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#This is the first review, where each word is converted to an integer
print(train_data[0])

#Reviews will be different lengths, we'll have to deal with that
print(len(train_data[0]), len(train_data[1]))

#This function converts integers back to words
word_index = imdb.get_word_index() #the dictionary mapping words to integer index

#reserves first 3 indices
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNKNOWN>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
    print(' '.join([reverse_word_index.get(i, '?') for i in text]))
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#Prints the review to console... See why we needed to reserve the first 3 indices?
decode_review(train_data[0])

#Data must be converted to tensors in order to feed the neural network
#Arrays will be padded to all have the same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=256)

#reviews now have 256 words, with 0s representing padding
print(len(train_data[0]),len(train_data[1]))
print(train_data[0])

#We want the model to predict either 0 (negative review) or 1 (positive review)
#How many layers do we need to do this?
vocab_size = 10000

model = keras.Sequential()

#Embedding can only be first layer
#turns positive integers into dense vectors of fixed size
#This layer takes the integer-encoded vocabulary and looks up corresponding vector for each word index
#These vectors are learned as the model trains
#Vectors are 3D, adding a dimension to the output array (I think that's what 16 represents?)
model.add(keras.layers.Embedding(vocab_size, 16)) 

#GlobalAveragePooling returns a fixed length, 1D average of the vectors for each example
#Averages over sequence dimension (16?)
#This lets the model handle input of variable length and return consistent data
model.add(keras.layers.GlobalAveragePooling1D())

#Fixed length output vector from above piped through Dense layer with 16 hidden units (neurons)
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

#Using sigmoid activation function, last layer outputs a number between 0 and 1 representing confidence interval/probability
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#So how many layers did we need? 4
#2 are input and output, and 2 are hidden
#More hidden layers means more freedom for the model to learn (more neurons) when learning an internal representation
#However, more layers means more computing power needed
#More layers may also lead to overfitting (patterns that apply to training data but not real data)

print("------------------------------------------")

#Now we need a loss function and optimizer for training
#Because model outputs a probability between 0 and 1, we use binary function
#binary_crossentropy measures "distance" between probability distributions
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
#remember metrics judges success by how accurate the prediction is

#We want to check the accuracy of the model on new data by creating a validation set
#Set aside 10,000 examples from original training data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#TRAIN THE MODEL!!! WOO!
#Trains 40 times over samples in x_train and y_train
#in batches of 512 samples
#we'll also monitor the model's loss and accuracy on the 10,000 samples from the validation set
history = model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)

#Evaluate model's performance
#2 values returned: Loss (error, low numbers are better) and Accuracy
results = model.evaluate(test_data,test_labels)
print("Evaluation Results: ",results)
#40 epochs gets about 87% accuracy

#Create graph of accuracy and loss over time
#model.fit() returns a history object with a dictionary with everything that happened during training
history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

#'bo' is for blue dot
plt.plot(epochs,loss,'bo',label='Training loss')
# 'b' is for solid blue line
plt.plot(epochs, val_loss, 'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#Validation loss and accuracy start to taper off and become linear
#This means the training loss and accuracy became overfitted
#The model learns specifications that only apply to training data, not test data
#Overfitting could be prevented after 20 or so epochs... This can be done automatically with a callback