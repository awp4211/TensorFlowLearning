# -*- coding: utf-8 -*-

"""
A Multilayer Perceptron implementation example using TensorFlow Library
This example is using the MNIST database
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters

n_hidden1 = 256 # 1st layer number of features
n_hidden2 = 256 # 2nd layer number of features
n_input = 784   # MNIST data input(img shape 28*28)
n_classes = 10  # MNIST total classes 

# TensorFlow Graph input
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

# Create Model
def multilayer_perceptron(x,weights,biases):
    # Hidden layer with RELU activation
    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.relu(layer1)
    # Hidden layer with RELU activation
    layer2 = tf.add(tf.matmul(x,weights['h2']),biases['b2'])
    layer2 = tf.nn.relu(layer2)
    # Output layer with linear activiation
    out_layer = tf.matmul(layer2,weights['out'])+biases['out']
    return out_layer
    
# Store layers weight and bias
weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden1])),
    'h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    'out':tf.Variable(tf.random_normal([n_hidden2,n_classes]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden1])),
    'b2':tf.Variable(tf.random_normal([n_hidden2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
pred = multilayer_perceptron(x,weights,biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the varibales
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Training Cycle