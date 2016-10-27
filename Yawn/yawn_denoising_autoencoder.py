# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


"""
Take an input tensor and add uniform masking
x:tensor/placeholder
x_corrupt:50% of value corrupt
"""
def corrupt(x):
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

    
