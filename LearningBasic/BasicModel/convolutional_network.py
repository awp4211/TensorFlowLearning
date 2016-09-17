# -*- coding: utf-8 -*-

"""
A Convolutional Network implementation example using TensorFlow Library
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784
n_classes = 10
dropout = 0.75 # Dropout,probability to keep units

# TensorFlow Graph Input
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# Create some wrappers for simplicity
def conv2d(x,W,b,strides=1):
    # strides 卷积步长为1
    # padding 0边距为模板
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# Create model
def conv_net(x,weights,biases,dropout):
    # x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是
    # 灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x = tf.reshape(x,shape=[-1,28,28,1])
    # Convolution Layer1
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    # ============================= TODO ======================================
    # print("=====",type(conv1)) <class 'tensorflow.python.framework.ops.Tensor'>
    # print(conv1.get_shape())  [?,28,28,32]
    # print(conv1.dtype) float32
    conv1 = maxpool2d(conv1,k=2)
    # print conv1.get_shape() [?,14,14,32]
    # ============================== TODO =====================================
    # Convolution Layer2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    # print conv2.get_shape() [?,14,14,64]
    conv2 = maxpool2d(conv2,k=2)
    # print conv2.get_shape() [?,7,7,64]
    # Fully connected layer
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1,dropout)
    # Output
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out
    
weights = {
    # 5 * 5 conv,1 input ,32 outputs
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    # 5 * 5 conv,32 inputs,64 outputs
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
     # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))