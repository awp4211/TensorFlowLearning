# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# Parameter
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameter
n_inputs = 28
n_steps = 28
n_hidden = 128
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
# TensorFlow LSTM cell reuqires 2*n_hidden length(state&cell)
istate_fw = tf.placeholder(tf.float32,[None,2*n_hidden])
istate_bw = tf.placeholder(tf.float32,[None,2*n_hidden])
y = tf.placeholder(tf.float32,[None,n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because forward + backword cells
    'hidden':tf.Variable(tf.random_normal([n_inputs,2*n_hidden])),
    'out':tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
}
biases = {
    'hidden':tf.Variable(tf.random_normal([2*n_hidden,])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(_x,
          _istate_fw,   # forward direction cell initial state
          _istate_bw,   # backword direction cell initial state
          _weights,
          _biases,      
          _batch_size,  #batch_size
          _seq_len      # n_steps
          ):
    # BiRNN requires to supply sequence_length as [batch_size,int64]    
    _seq_len = tf.fill([_batch_size],tf.constant(_seq_len,dtype=tf.int64))
    
    # input shape[batch_size,n_steps,n_inputs] ==> [n_steps,batch_size,n_inputs]
    _x = tf.transpose(_x,[1,0,2])
    # reshape to [n_steps * batch_size,n_inputs]
    _x = tf.reshape(_x,[-1,n_inputs])
    
    # Linear activation
    _x = tf.matmul(_x,_weights['hidden']) + _biases['hidden']
    
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    # split data because rnn cell needs a list of inputs for RNN inner loop
    # ==> n_steps * [batch_size,n_hidden]
    _x = tf.split(0,n_steps,_x)
    
    outputs = tf.nn.bidirectional_rnn(
        lstm_fw_cell,
        lstm_bw_cell,
        _x,
        initial_state_fw=_istate_fw,
        initial_state_bw=_istate_bw,
        sequence_length=_seq_len
    )
    
    # Linear activation
    return tf.matmul(outputs[-1],_weights['out']) + _biases['out']


pred = BiRNN(x,istate_fw,istate_bw,weights,biases,batch_size,n_steps)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Initializing the variabels
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    best_acc = 0.
    
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        # reshape data to get 28 seq of 28 elements ==> (batch_size,n_step,n_inputs)
        batch_xs = batch_xs.reshape((batch_size,n_steps,n_inputs))
        # Training using batch data
        sess.run(optimizer,feed_dict={
            x:batch_xs,
            y:batch_ys,
            istate_fw:np.zeros((batch_size,2*n_hidden)),
            istate_bw:np.zeros((batch_size,2*n_hidden))
        })
        
        if step % display_step == 0:
            # Calculate batch accuracy and loss
            acc,loss = sess.run([accuracy,cost],feed_dict={
                x: batch_xs, 
                y: batch_ys,
                istate_fw: np.zeros((batch_size, 2*n_hidden)),
                istate_bw: np.zeros((batch_size, 2*n_hidden))
            })
            if acc > best_acc:
                best_acc = acc
            print("Iter {0},Minibatch Loss = {1},Training Accuracy = {2}".format(
                step*batch_size,loss,acc))
        step += 1
    print("Optimization Finished!")
    print("Best Accuracy:{0}".format(best_acc))
    
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_inputs))
    test_label = mnist.test.labels[:test_len]
    
    print("Testing Accuracy:{0}".format(
        sess.run(accuracy,feed_dict={
            x:test_data,
            y:test_label,
            istate_fw:np.zeros((test_len,2*n_hidden)),
            istate_bw:np.zeros((test_len,2*n_hidden))
        })        
    ))
