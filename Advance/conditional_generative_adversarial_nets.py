# -*- coding: utf-8 -*-

"""
Conditional Generative Adversarial Nets
paper:Conditional Generative Adversarial Nets.arXiv:1411.1784v1
This is the implementation of CGAN for MNIST_DATA
learned from https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow_v1.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# parameter
z_dim = 100
x_dim = 28*28
y_dim = 10
h_dim = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def generator(z, y):
    """
    Generator NETS:
        The generator(z,y) takes both of z_dim-dimensional vector and label of data
        and returns x_dim-dimensional vector, which is MNIST image(28*28).
        z here is the prior for the G(z).In a way it learns a mapping between the prior space to Pdata.
    :param z:
    :param y:
    :return:
    """
    inputs = tf.concat(axis=1, values=[z, y])
    g_w1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
    g_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    g_h1 = tf.nn.relu(tf.matmul(inputs, g_w1) + g_b1)

    g_w2 = tf.Variable(xavier_init([h_dim, x_dim]))
    g_b2 = tf.Variable(tf.zeros(shape=[x_dim]))
    g_log_prob = tf.matmul(g_h1, g_w2) + g_b2

    g_prob = tf.nn.sigmoid(g_log_prob)

    theta_g = [g_w1, g_b1, g_w2, g_b1]

    return g_prob, theta_g


d_w1 = tf.Variable(xavier_init([x_dim + y_dim, h_dim]))
d_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
d_w2 = tf.Variable(xavier_init([h_dim, 1]))
d_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_d = [d_w1, d_b1, d_w2, d_b2]


def discriminator(x, y):
    """
    Discriminator NETS
    :param x:
    :param y:
    :return:
    """
    inputs = tf.concat(axis=1, values=[x, y])
    
