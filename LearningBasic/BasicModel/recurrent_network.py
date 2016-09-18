# -*- coding: utf-8 -*-
"""
A Recurrent Neural Network(LSTM) implementation example using TensorFlow Library
This example is using the MNIST databasee of handwritten digits
"""

import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
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
    # Prepare data shape to match 'rnn' function requirements
    # Current data input shape:(batch_size,n_steps,n_input)
    # Require shape: 'n_steps':tensors list of shape(batch_size,n_input)

    # Premuting(转置) batch_szie and n_steps -->(n_steps,batch_size,n_input)
    x = tf.transpose(x,[1,0,2])
    # Reshaping to (n_steps*batch_size,n_input)
    x = tf.reshape(x,[-1,n_input])
    # Split to get a list of 'n_steps' tensors of shape(batch_size,n_input)
    x = tf.split(0,n_steps,x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    
    # Get lstm cell output
    outputs,states = rnn.rnn(lstm_cell,x,dtype=tf.float32)
    
    # Linear activation,using rnn inner loop last output
    return tf.matmul(outputs[-1],weights['out']) + biases['out']

# Define loss and optimizer
pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 sequences of 28 elements
        batch_x = batch_x.reshape((batch_size,n_steps,n_input))
        sess.run(optimizer,feed_dict={x:batch_x,
                                      y:batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy,feed_dict={x:batch_x,
                                               y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,
                                            y:batch_y})
            print("Iter {0} Minibatch Loss = {1} ,Training Accuracy = {2}".format(
                        step*batch_size,loss,acc))
        step+=1
    print("Optimization Done")
    
    # Calculate accuracy for 128 mnist test images
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    test_acc = sess.run(accuracy,feed_dict={x: test_data, 
                                            y: test_label})
    print("Testing Accuracy:{0}".format(test_acc))





















    