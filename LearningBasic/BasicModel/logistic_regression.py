# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Parameters

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tensorflow graph input
x = tf.placeholder(tf.float32,[None,784])# mnist data 
y = tf.placeholder(tf.float32,[None,10])# 10 classes

# set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x,W)+b)

# Minimize error using coree entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # print(type(batch_xs)) <type 'numpy.ndarray'>
            # print(batch_xs.shape) (100,784)
            # Run optimization op(backprop) and cost op(to get loss value)
            _, c = sess.run([optimizer,cost],feed_dict={x:batch_xs,
                                                     y:batch_ys})
            # Compute average loss
            avg_cost += c/total_batch
        if (epoch+1) % display_step == 0:
            _str = "Epoch: {0},cost = {1}".format(epoch,avg_cost)
            print(_str)

    print("Optimization Finished!")
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))