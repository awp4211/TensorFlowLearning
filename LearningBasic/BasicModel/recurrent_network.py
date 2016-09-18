# -*- coding: utf-8 -*-
"""
A Recurrent Neural Network(LSTM) implementation example using TensorFlow Library
This example is using the MNIST databasee of handwritten digits
"""

import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

"""
To classify images using a recurrent neural network,we consider every image 
row as a sequecne of pixels.Because MNIST image shape is 28*28px,we will then
handle 28 sequences of 28 steps for every sample
"""

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input(img shape:28*28)
n_steps = 28 # Timesteps
n_hidden = 128 # Hidden layer num of features
n_classes = 10 # MNIST total classes

# TensorFlow Graph input
x = tf.placeholder("float",[None,n_steps,n_input])
y = tf.placeholder("float",[None,n_classes])

# Define weights
weights = {
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x,weights,biases):
    # Prepare data shape to match 'rnn'