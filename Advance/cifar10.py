# -*- coding: utf-8 -*-

import gzip
import os
import re
import sys
import tarfile

import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input


# Basic model parameters
batch_size = 128
data_dir = "/CIFAR10_data"

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _activation_summary(x):
    """
    Create summaries for activations
    Creates a summary that provides a histogram of activations
    Creates a summary that measures the sparsity of activations
    x:Tensor
    """
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations',x)
    tf.scalar_summary(tensor_name + '/sparsity',tf.nn.zero_fraction(x))
    
def inference(images):
    """
    Build the CIFAR-10 Model
    """