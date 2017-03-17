# -*- coding: utf-8 -*-

"""
This is the implementation of pix2pix
paper:Image-to-Image Translation with Conditional Adversarial Networks
"""

import tensorflow as tf
import numpy as np
import os
import json
import glob
import random
import collections
import time
import math


def download_dataset():
    """
    Downloading facades dataset
    :return:
    """
    import sys
    import tarfile
    import tempfile
    import shutil
    from urllib2 import urlopen  # python 2
    url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % 'facades'
    with tempfile.TemporaryFile() as tmp:
        print("downloading", url)
        shutil.copyfileobj(urlopen(url), tmp)
        print("extracting")
        tmp.seek(0)
        tar = tarfile.open(fileobj=tmp)
        tar.extractall()
        tar.close()
        print("done")


def preprocess(image):
    """
    Translate image data from [0, 1] to [-1, 1]
    :param image:
    :return:
    """
    with tf.name_scope("preprocess"):
        #[0, 1] => [-1, 1]
        return image * 2 - 1


def preprocess_lab(lab):
    """

    :param lab:
    :return:
    """
    with tf.name_scope("preprocess_lab"):
        pass


def deprocess_lab():
    pass


def augment(image,brightness):
    pass


def conv2d(batch_input, output_channels, stride):
    """
    Convolution Operator of tensors
    :param batch_input: input_tensor,shape=[batch_size, in_height, in_weight, in_channels]
    :param output_channels:
    :param stride:
    :return:
    """
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter",[4, 4, in_channels, output_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0,0.02))
        #[batch_size, in_height, in_width, in_channels],
        #[filter_width, filter_height, in_channels, out_channels]
        #==> [batch_size, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input,[[0, 0], [1, 1], [1, 1], [0,0]], mode='CONSTANT')
        conv2d = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding='VALID')
        return conv2d


def lrelu(x, a):
    """
    Adding these together creates the leak part and linear part.
    Then cancels them out by subtracting/adding an absolute value term
        LEAK: a*x /2 - a*abs(x)/2
        LINEAR: x/2 + abs(x)/2
    :param x:
    :param a:
    :return:
    """
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    """
    This block looks like it has 3 inputs on the graph unless we do this
    :param input:
    :return:
    """
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)
        channels = input.get_shape()[-1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input,
                    mean, variance, offset, scale,
                    variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    """
    Deconvolution
    :param batch_input:
    :param out_channels:
    :return:
    """
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels],
                                 dtype=tf.float32, initializer=tf.random_uniform_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels],
        # [filter_width, filter_height, out_channels, in_channels]
        # => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter,
                                      [batch, in_height*2, in_width*2, out_channels],
                                      [1, 2, 2, 1],
                                      padding='SAME')
        return conv


def check_image(image):
    pass


def rgb_to_lab(srgb):
    pass


def lab_to_rgb(lab):
    pass


def load_examples():
    pass


def create_generator(generator_inputs,
                     generator_outputs_channels,
                     ngf=64):
    """
    Generator Nets
    :param generator_inputs:
    :param generator_outputs_channels:
    :param nfg:number of generator filters in first conv layer
    :return:
    """
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, nfg]
    with tf.variable_scope("encoder_1"):
        output = conv2d(generator_inputs, ngf, stride=2)
        layers.append(output)

    layer_specs = [
        ngf * 2,#encoder_2:[batch, 128, 128, ngf] => [batch, 64, 64, ngf*2]
        ngf * 4,#encoder_3:[batch, 64, 64, ngf*2] => [batch, 32, 32, ngf*4]
        ngf * 8,#encoder_4:[batch, 32, 32, ngf*4] => [batch, 16, 16, ngf*8]
        ngf * 8,#encoder_5:[batch, 16, 16, ngf*8] => [batch, 8, 8, ngf*8]
        ngf * 8,#encoder_6:[batch, 8, 8, ngf*8] => [batch, 4, 4, ngf*8]
        ngf * 8,#encoder_7:[batch, 4, 4, ngf*8] => [batch, 2, 2, ngf*8]
        ngf * 8 #encoder_8:[batch, 2, 2, ngf*8] => [batch, 1, 1, ngf*8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) +1 )):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height/2, in_width/2, out_channels]
            conv = conv2d(rectified, out_channels, stride=2)
            output = batchnorm(conv)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                intput = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels]
            # ==>[batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        pass

