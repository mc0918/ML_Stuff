# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:29:53 2019

@author: mcamp

https://www.tensorflow.org/tutorials/keras/basic_regression
"""

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Download dataset and save it as dataset_path variable
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

#Use pandas to import the data
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

#Clean the dataset by dropping unknown values
dataset.isna().sum()
dataset = dataset.dropna()

#Because "Origin" column is categorical and not numeric,
#we need to encode it by converting it to a one-hot
origin = dataset.pop('Origin')

#We set values to correspond to unique Origins
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

#Split data into test and training set
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Now take a look at some column pairs from the dataset
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

#And a look at the overall statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

#Now we need to split the target value, "label", from the features. 
#the label is the value the model is being trained to predict
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#The data/features need to be normalized. 
#the model might work with different ranges/scales but at best it will then be dependent on the units you use
#the normalized data is used to train the model
#The statistics used to normalize must be used with EVERY other data fed to the model, ALONG with the one-hot encoding
def norm(x):
    return(x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

## NOW WE ACTUALLY BUILD THE MODEL ##
#We're using a Sequential model, with 2 densely connected layers and 1 output layer
#the output layer returns a single, continuous value
def build_model():
    model = keras.Sequential([
            layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_dataset.keys())]),
            layers.Dense(64, activation = tf.nn.relu),
            layers.Dense(1)
            ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                  optimizer = optimizer,
                  metrics = ['mean_absolute_error','mean_squared_error'])
    
    return model

#Run the model and print a simple description
model = build_model()
model.summary()

#now try out the model, taking a batch of 10 examples from training data
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print("Example result: ", example_result)

## TRAIN THE MODEL ##
#this class just displays progress by printing a dot for each epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

#Visualize the model's training progress using history object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

#After about 100 epochs, the model doesn't improve
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

#To improve the model we add an EarlyStopping callback function
#If a set amount of epochs passes without improving the model, the training stops
model = build_model()

#The patience parameter is the amount of epochs to check for improvement
#With this, average error is about +/- 2 MPG
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

#Now we see how well the model generalizes, using the test dataset
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
#Cool, it pretty much works

#But can it make predictions accurately?
test_predictions = model.predict(normed_test_data).flatten()

#plt.scatter(test_labels, test_predictions)
#plt.xlabel('True Values [MPG]')
#plt.ylabel('Predictions [MPG]')
#plt.axis('equal')
#plt.axis('square')
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])

#Reasonable prediction, now let's see error
#Comment out lines 171-178 to get a new error plot
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
_ = plt.plot()
#Almost a Gaussian distribution, if we increase the sample size we'd probably be happy
