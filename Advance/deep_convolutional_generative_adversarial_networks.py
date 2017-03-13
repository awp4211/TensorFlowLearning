# -*- coding: utf-8 -*-


"""
implement of DCGAN(Deep Convolutional Generative Adversarial Networks)
Unsupervised representation learning with deep
convolutional generative adversarial networkss

"""
import math
import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope



def encoder(input_tensor, output_size):
    """
    Create encoder network
    :param input_tensor:a batch of flattenen images [batch_size, 28*28]
    :param output_size:
    :return:A tensor that expresses the encoder network
    """
    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    print('encoder network -- reshape,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d(net, 32, 5, stride=2)
    print('encoder network -- conv1,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d(net, 64, 5, stride=2)
    print('encoder network -- conv2,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d(net, 128, 5, stride=2,padding='VALID')
    print('encoder network -- conv3,shape = {0}'.format(net.get_shape()))

    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    print('encoder network -- flatten,shape = {0}'.format(net.get_shape()))

    net = layers.fully_connected(net, output_size, activation_fn=None)
    print('encoder network -- FC,shape = {0}'.format(net.get_shape()))

    return net


def discriminator(input_tensor):
    """
    Create a network that discriminates between images from a dataset and
    generated ones.
    :param input_tensor:a batch of real images[batch, height, width, channels]
    :return:a tensor that represents the network
    """
    return encoder(input_tensor, 1)

def decoder(input_tensor):
    """
    Create decoder network
    If input tensor is provided then decodes it,
        otherwise samples from a sampled vector
    :param input_tensor: a batch of vector to decode
    :return: a tensor that expresses the decoder network
    """
    print('decoder network -- input_tensor,shape = {0}'.format(input_tensor.get_shaoe()))
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    print('decoder network -- expand_dims,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
    print('decoder network -- dconv1,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
    print('decoder network -- dconv2,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 32, 5, stride=2)
    print('decoder network -- dconv3,shape = {0}'.format(net.get_shape()))

    net = layers.conv2d_transpose(
        net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid
    )
    print('decoder network -- dconv4,shape = {0}'.format(net.get_shape()))

    net = layers.flatten(net)
    print('decoder network -- flatten,shape = {0}'.format(net.get_shape()))

    return net


