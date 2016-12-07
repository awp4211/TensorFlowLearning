# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

n_samples = mnist.train.num_examples

def xavier_init(fan_in,fan_out,constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                             minval=low,maxval=high,
                             dtype=tf.float32)
                             
class VaritaonalAutoencoder(object):
    # tf.nn.softplus : log(exp(features) + 1)
    def __init__(self,
                 network_architecture,
                 transfer_fct=tf.nn.softplus,
                 learning_rate=0.001,
                 batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32,[None,network_architecture["n_input"]])
        # create autoencoder network
        self._create_network()
        # define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()
        
        # initializing the tensorflow variables
        init = tf.initialize_all_variables()
        
        # launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def _create_network(self):
        network_weights = self._initialize_weights(**self.network_architecture)