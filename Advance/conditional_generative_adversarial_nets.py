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

from tensorflow.examples.tutorials.mnist import input_data

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
    d_h1 = tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1)
    d_logit = tf.matmul(d_h1, d_w2) + d_b2
    d_prob = tf.nn.sigmoid(d_logit)

    return d_prob, d_logit


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train(batch_size=128,
          learning_rate=0.0001):
    print('...... loading dataset ......')
    mnist = input_data.read_data_sets('MNIST/', one_hot=True)

    print('...... define model ......')
    x = tf.placeholder(tf.float32, shape=[None, x_dim])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    g_sample, theta_g = generator(z, y)
    d_real, d_logit_real = discriminator(x, y)
    d_fake, d_logit_fake = discriminator(g_sample, y)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

    d_solver = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=theta_d)
    g_solver = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=theta_g)

    if not os.path.exists('CGAN_OUT/'):
        os.makedirs('CGAN_OUT/')

    print('...... start to training ......')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    index = 0
    for epoch in range(1000000):
        if epoch % 1000 == 0:
            n_sample = 16
            z_sample = sample_z(n_sample, z_dim)
            # sample data z and lable y
            y_sample = np.zeros(shape=[n_sample, y_dim])
            y_sample[:, 7] = 1# set all labels to 8

            samples = sess.run(g_sample,
                               feed_dict={z: z_sample,
                                          y: y_sample})
            fig = plot(samples)
            plt.savefig('CGAN_OUT/{0}.png'.format(str(index).zfill(3)), bbox_inches='tight')
            index = index + 1

        xs, ys = mnist.train.next_batch(batch_size)
        z_sample = sample_z(batch_size, z_dim)
        _, d_loss_curr = sess.run([d_solver, d_loss],
                                  feed_dict={x: xs, y: ys, z: z_sample})
        _, g_loss_curr = sess.run([g_solver, g_loss],
                                  feed_dict={z: z_sample, y: ys})

        if epoch % 1000 == 0:
            print('---epoch:{0}'.format(epoch))
            print('d_loss:{:.4}'.format(d_loss_curr))
            print('g_loss:{:.4}'.format(g_loss_curr))
            print('---------------------------------')


if __name__ == '__main__':
    train()

