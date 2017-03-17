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


def deprocess_lab():
    pass


def augment(image,brightness):
    pass


def conv(batch_input, output_channels, stride):
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
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding='VALID')
        return conv


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
        offset = tf.get_variable("offset", [channels], dtype=tf.float32)
