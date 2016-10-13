# -*- coding: utf-8 -*-

# TODO
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# HyperParameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# Network Parameter
n_input = 784
n_classes = 10
dropout = 0.8

# Input
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# Convolution
def conv2d(name,l_input,w,b):
    conv = tf.nn.conv2d(l_input,w,strides=[1,1,1,1],padding="SAME")
    return tf.nn.relu(tf.nn.bias_add(conv,b),name=name)
    
# Max Pool
def max_pool(name,l_input,k):
    return tf.nn.max_pool(l_input,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME",name=name)
    
def alex_net():
    pass
