# -*- coding: utf-8 -*-


"""
implement of DCGAN(Deep Convolutional Generative Adversarial Networks)
Unsupervised representation learning with deep
convolutional generative adversarial networks

TODO DEBUG
https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW/blob/master/
"""
import math
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from progressbar import ETA, Bar, Percentage, ProgressBar


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

    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
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
    print('decoder network -- input_tensor,shape = {0}'.format(input_tensor.get_shape()))
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


def concat_elu(inputs):
    print(inputs.get_shape())

    return tf.nn.elu(tf.concat(3, [-inputs, inputs]))


class Generator(object):

    def update_params(self, input_tensor):
        """
        update parameters of the network
        :param input_tensor:a batch of flattened images
        :return:current loss value
        """
        return NotImplementedError()

    def generate_and_save_images(self, num_samples, directory):
        """
        generates the images using the model and saves them in the directory
        :param num_samples: number of samples to generate
        :param directory:
        :return:
        """
        imgs = self.sess.run(self.sampled_tensor)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'dcgan_imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder, '%d.png') % k,
                   imgs[k].reshape(28, 28))


class GAN(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate):
        self.input_tensor = tf.placeholder(tf.float32, [None, 28*28])

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True}):
            with tf.variable_scope('model'):
                D1 = discriminator(self.input_tensor)# postive examples
                D_params_sum = len(tf.trainable_variables())
                G = decoder(tf.random_normal([batch_size, hidden_size]))
                self.sampled_tensor = G

            with tf.variable_scope('model', reuse=True):
                D2 = discriminator(G)# generated examples

        D_loss = self.__get_discriminator_loss(D1, D2)
        G_loss = self.__get_generator_loss(D2)

        params = tf.trainable_variables()
        D_params = params[:D_params_sum]
        G_params = params[D_params_sum:]
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_discrimator = layers.optimize_loss(
                D_loss, global_step, learning_rate / 10, 'Adam', variables=D_params, update_ops=[])
        self.train_generator = layers.optimize_loss(
                G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __get_discriminator_loss(self, D1, D2):
        """
        Loss for the discriminator network
        :param D1:logits computed with a discriminator networks from real images
        :param D2:logits computed with a discriminator networks from generated images
        :return:Cross entropy loss, postive samples have implicit labels 1, negative 0ss
        """
        return (losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1))) +
                losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D1))))

    def __get_generator_loss(self, D2):
        """
        Loss for generator.Maximize probability of generating images that
            discrimator cannot differentiate.
        :param D2:
        :return:
        """
        return losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2)))

    def update_params(self, inputs):
        d_loss_value = self.sess.run(self.train_discrimator, {self.input_tensor, inputs})

        g_loss_value = self.sess.run(self.train_generator)

        return g_loss_value


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    # parameter
    hidden_size = 128
    batch_size = 128
    max_epoch = 100
    learning_rate = 1e-2
    updates_per_epoch = 1000
    mnist = input_data.read_data_sets('MNIST/', one_hot=True)

    model = GAN(hidden_size, batch_size, learning_rate)
    for epoch in range(max_epoch):
        training_loss = 0.0
        pbar = ProgressBar()
        for i in pbar(range(updates_per_epoch)):
            images,_ = mnist.train.next_batch(batch_size)
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / (updates_per_epoch * batch_size)
        print("Loss %f" % training_loss)
        model.generate_and_save_images(batch_size,"GAN_IMG/")