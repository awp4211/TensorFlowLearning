# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

"""
Take an input tensor and add uniform masking
x:tensor/placeholder
x_corrupt:50% of value corrupt
"""
def corrupt(x):
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

"""
Build a deep denoising autoencoder
parameters:
    width,height:input dimension
    dimensions:list of DAE dimension
"""
def autoencoder(x,width=256,height=256,
                dimensions=[784,512,256,64]):
    x = tf.placeholder(tf.float32,[None,dimensions[0]])
    corrupt_prob = tf.placeholder(tf.float32,[1])
    
    # current layer input
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)
    
    print('...... building the encoder ......')
    encoder = []
    for layer_i,n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape().as_list()[1])
        W = tf.Variable(
                tf.random_uniform([n_input,n_output],
                                  -1.0/math.sqrt(n_input),
                                   1.0/math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input,W)+b)
        current_input = output
    
    # latent representation
    z = current_input
    
    
    encoder.reverse()
    for layer_i,n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input,W)+b)
        current_input = output
        
    # now have the recontruction through the network
    y = current_input    
    # cost function measures pixel-wise difference
    