# -*- coding: utf-8 -*-

"""
This is a test program to know how mnist_data prepare to learn
"""

from tensorflow.examples.tutorials.mnist import input_data
batch_size = 20
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


batch_xs,batch_ys = mnist.train.next_batch(batch_size)


# shape test
print batch_xs.shape#(batch_size,784)
print batch_ys.shape#(batch_size,10)

# value test
print batch_xs[0]#[0.xxx,0.xxx......0.xxxx]
print batch_ys[0]#[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
