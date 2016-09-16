# -*- coding: utf-8 -*-

# nearest neighbor learning algorithm example using TensorFlow Library

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# In thsi example,we limit mnist data
Xtr,Ytr = mnist.train.next_batch(5000)
Xte,Yte = mnist.test.next_batch(200)

# tf Graph Input
xtr = tf.placeholder("float",[None,784])
xte = tf.placeholder("float",[784])

# Nearest Neighbor L1 Distance
distance = tf.reduce_sum(tf.abs(
                            tf.add(xtr,tf.neg(xte))
                            ),
                            reduction_indices=1)
                            
# Prediction:Get min distance index
pred = tf.arg_min(distance,0)

accuracy = 0

# Initializing the varibales
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(len(Xte)):
        # Get nearest neighbor        
        nn_index = sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        # Get nearest neighbor class label and compare it to its true label
        print "Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i])
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print "Done!"
    print "Accuracy:", accuracy