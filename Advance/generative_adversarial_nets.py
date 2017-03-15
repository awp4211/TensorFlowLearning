# -*- coding: utf-8 -*-

"""
Generative Adversarial Nets(GAN)
paper:Generative Adversarial Nets.arxiv:1406.2661v1
This is the implementation of GAN for MNIST dataset
learned from:https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow_v1.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim /2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def generator(z):
    """
    Generator Net
    :param z:input noise sampled from N(0,1)
    :return:generated data from noise
    """
    g_w1 = tf.Variable(xavier_init([100, 128]))
    g_b1 = tf.Variable(tf.zeros(shape=[128]))
    g_h1 = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)

    g_w2 = tf.Variable(xavier_init([128, 784]))
    g_b2 = tf.Variable(tf.zeros(shape=[784]))
    g_log_prob = tf.matmul(g_h1, g_w2) + g_b2

    g_prob = tf.nn.sigmoid(g_log_prob)

    theta_g = [g_w1, g_w2, g_b1, g_b2]

    return g_prob, theta_g


d_w1 = tf.Variable(xavier_init([784, 128]))
d_b1 = tf.Variable(tf.zeros(shape=[128]))
d_w2 = tf.Variable(xavier_init([128, 1]))
d_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_d = [d_w1, d_w2, d_b1, d_b2]

def discriminator(x):
    """
    Discriminator Net
    :param x:input data
    :return:is data from x(d_prob) and log likelihood
    """
    d_h1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    d_logit = tf.matmul(d_h1, d_w2) + d_b2
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def sample_z(m,n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train(batch_size=128,
          learning_rate=0.001):

    learning_rate = 0.001

    print('...... define model ......')
    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784])
    z_dim = 100
    z = tf.placeholder(tf.float32, shape=[None, 100])#z_dim = 100

    g_sample, theta_g = generator(z)
    d_real, d_logit_real = discriminator(x)
    d_fake, d_logit_fake = discriminator(g_sample)

    # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    # G_loss = -tf.reduce_mean(tf.log(D_fake))

    # define loss
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

    d_solver = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=theta_d)
    g_solver = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=theta_g)

    print('...... loading dataset......')
    mnist = input_data.read_data_sets('MNIST/', one_hot=True)

    if not os.path.exists('GAN_OUT/'):
        os.makedirs('GAN_OUT/')

    print('...... start to training ......')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    index = 0
    for epoch in range(1000000):
        if epoch % 1000 == 0:
            samples = sess.run(g_sample,
                               feed_dict={z: sample_z(16, z_dim)})
            fig = plot(samples)
            plt.savefig('GAN_OUT/{0}.png'.format(str(index).zfill(3)), bbox_inches='tight')
            index = index + 1
            plt.close(fig)

        xs, _ = mnist.train.next_batch(batch_size)

        _, d_loss_curr = sess.run([d_solver, d_loss],
                                  feed_dict={x: xs,
                                             z: sample_z(batch_size, z_dim)})
        _, g_loss_curr = sess.run([g_solver, g_loss],
                                  feed_dict={z: sample_z(batch_size, z_dim)})

        if epoch % 1000 == 0:
            print('---epoch:{0}'.format(epoch))
            print('d_loss:{:.4}'.format(d_loss_curr))
            print('g_loss:{:.4}'.format(g_loss_curr))
            print('---------------------------------')


if __name__ == '__main__':
    train()




