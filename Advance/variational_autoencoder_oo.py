# -*- coding: utf-8 -*-

"""
OO styled Variational Autoencoder
"""

import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope


def encoder(input_tensor, output_size):
    """
    Create encoder network
    :param input_tensor:a batch of flattened images [batch_size,28*28]
    :param output_size:return size
    :return: encodered neural networks
    """
    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    net = layers.conv2d(net, num_outputs=32, kernel_size=5, stride=2, padding='SAME')
    net = layers.conv2d(net, num_outputs=64, kernel_size=5, stride=2, padding='SAME')
    net = layers.conv2d(net, num_outputs=128, kernel_size=5, stride=2, padding='VALID')
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, output_size, activation_fn=None)


def decoder(input_tensor):
    """
    Create decoder network.If input is provided then decodes it, otherwise samples from a vector.
    :param input_tensor: a batch of vectors to decode
    :return: A tensor that expresses the decoder network
    """
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)


