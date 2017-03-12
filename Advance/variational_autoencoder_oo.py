# -*- coding: utf-8 -*-

"""
OO styled Variational Autoencoder
"""

import numpy as np
import tensorflow as tf
import os

from scipy.misc import imsave
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from progressbar import ETA, Bar, Percentage, ProgressBar


def encoder(input_tensor, output_size):
    """Create encoder network.
    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
    """
    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    net = layers.conv2d(net, 32, 5, stride=2)
    net = layers.conv2d(net, 64, 5, stride=2)
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, output_size, activation_fn=None)


def decoder(input_tensor):
    """
        Create decoder network.
        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode
    Returns:
        A tensor that expresses the decoder network
    """

    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
    net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
    net = layers.conv2d_transpose(net, 32, 5, stride=2)
    net = layers.conv2d_transpose(
        net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
    net = layers.flatten(net)
    return net


class Generator(object):

    def update_params(self, input_tensor):
        """
        Update parameters of the network
        :param input_tensor:
        :return:
        """
        raise NotImplementedError()

    def generate_and_save_images(self, num_samples, directory):
        """
        Generates the image using the model and saves them in the directory
        :param num_samples:num_samples:number of samples to generate
        :param directory:a directory to save the images
        :return:
        """
        imgs = self.sess.run(self.sampled_tensor)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)
            imsave(os.path.join(imgs_folder, '%d.png') % k, imgs[k].reshape(28,28))


class VAE(Generator):

    def __init__(self,
                 hidden_size,
                 batch_size,
                 learning_rate):
        self.input_tensor = tf.placeholder(tf.float32, [None, 28*28])
        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale':True}):
            with tf.variable_scope('model') as scope:
                encoded = encoder(self.input_tensor, hidden_size * 2)
                mean = encoded[:, :hidden_size]
                stddev = tf.sqrt(tf.exp(encoded[:, hidden_size:]))
                epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])
                input_sample = mean + epsilon * stddev
                output_tensor = decoder(input_sample)

            with tf.variable_scope('model', reuse=True) as scope:
                # reuse model to sample from hidden layer
                self.sampled_tensor = decoder(tf.random_normal([batch_size, hidden_size]))

        vae_loss = self.__get_vae_cost(mean, stddev)
        rec_loss = self.__get_reconstruction_cost(output_tensor, self.input_tensor)

        loss = vae_loss + rec_loss
        self.train = layers.optimize_loss(loss,
                                          tf.contrib.framework.get_or_create_global_step(),
                                          learning_rate=learning_rate,
                                          optimizer='Adam',
                                          update_ops=[])
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __get_vae_cost(self, mean, stddev, epsilon=1e-8):
        """
        vae loss
        :param mean:
        :param stddev:
        :param epsilon:
        :return:
        """
        return tf.reduce_sum(0.5*(tf.square(mean) + tf.square(stddev) -
                                  2.0 * tf.log(stddev + epsilon) - 1.0))

    def __get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        """
        reconstruction loss
        :param output_tensor:tensor produces by decoder
        :param target_tensor: the target tensor that we want to reconstruct
        :param epsilon:
        :return:
        """
        return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                             (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def update_params(self, input_tensor):
        """
        update parameters of the network
        :param input_tensor:
        :return:
        """
        return self.sess.run(self.train,{self.input_tensor: input_tensor})

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    # parameter
    hidden_size = 128
    batch_size = 128
    max_epoch = 100
    learning_rate = 1e-2
    updates_per_epoch = 1000
    mnist = input_data.read_data_sets('MNIST/', one_hot=True)

    model = VAE(hidden_size, batch_size, learning_rate)
    for epoch in range(max_epoch):
        training_loss = 0.0
        pbar = ProgressBar()
        for i in pbar(range(updates_per_epoch)):
            images,_ = mnist.train.next_batch(batch_size)
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / (updates_per_epoch * batch_size)
        print("Loss %f" % training_loss)
        model.generate_and_save_images(batch_size,"IMG/")