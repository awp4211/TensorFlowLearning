# -*- coding: utf-8 -*-


"""
Stack denoising autoencoder

"""
import tensorflow as tf
import sys
import process_data as pd
import datetime
import math
import numpy as np

class dA(object):
    def __init__(self,net_input,n_input,n_hidden):
        self.w = tf.Variable(tf.random_uniform(
                        shape=[n_input,n_hidden],
                        minval=-4*np.sqrt(6./(n_input+n_hidden)),
                        maxval= 4*np.sqrt(6./n_input+n_hidden)))
        self.b_hidden = tf.Variable(tf.constant(0.0,shape=[n_hidden,]))
        self.b_input = tf.Variable(tf.constant(0.0,shape=[n_input,]))
        self.input = net_input
        
    def get_hidden_values(self,input):
        return tf.nn.sigmoid(tf.matmul(self.input,self.w)+self.b_hidden)
        
    def get_reconstructed_input(self,hidden):
        w_t = tf.transpose(self.w)
        return tf.nn.sigmoid(tf.matmul(hidden,w_t)+self.b_input)
        
    def get 