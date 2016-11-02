# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 500
corruption_level = 0.3

x = tf.placeholder(tf.float32,[None,n_visible])

# 用于将部分数据置0
mask = tf.placeholder(tf.float32,[None,n_visible])

# create nodes for hidden variables
w_init_max = 4 * np.sqrt(6./(n_visible + n_hidden))
w_init = tf.random_uniform(shape=[n_visible,n_hidden],
                           minval=-w_init_max,
                           maxval=w_init_max)

# encoder
w = tf.Variable(w_init,name='w')# shape:784 * 500
b = tf.Variable(tf.zeros([n_hidden]))# 隐含层偏置
# decoder
w_prime = tf.transpose(w)
b_prime = tf.Variable(tf.zeros([n_visible]))

def model(x,mask,w,b,w_prime,b_prime):
    tilde_x = mask * x # corrupted x
    y = tf.nn.sigmoid(tf.matmul( tilde_x, w) + b)
    z = tf.nn.sigmoid(tf.matmul( y, w_prime) + b_prime)
    return z

z = model(x,mask,w,b,w_prime,b_prime)

cost = tf.reduce_sum(tf.pow(x-z,2))
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={x: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={x: teX, mask: mask_np}))