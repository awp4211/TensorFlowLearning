# -*- coding: utf-8 -*-
"""
Using RNN models to predict continues Data like sine and cosine
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_start = 0
time_steps = 20
batch_size = 50
input_size = 1
output_size = 1
cell_size = 10
lr = 0.006

def get_batch():
    global batch_start,time_steps
    # xs shape(50batch,20steps)
    xs = np.arange(batch_start,batch_start + time_steps * batch_size)
    xs = xs.reshape((batch_size,time_steps)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += time_steps
    
    #================================DISPLAY=================================== 
    #plt.ion()
    #plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    #plt.show()
    #================================DISPLAY===================================    
    
    # returned seq res and xs:shape(batch,time_step,input)    
    # seq[:,:,np.newaxis] shape ==> (50,20,1)
    # res[:,:,np.newaxis] shape ==> (50,20,1)
    # xs shape ==> (50,20)
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]
    
class LSTMRNN(object):
    def __init__(self,
                 n_steps,# 序列长度
                 input_size,# 输入数据的维度
                 output_size,
                 cell_size,
                 batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            self.ys = tf.placeholder(tf.float32,[None,n_steps,output_size],name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)
            
    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs)
        pass
    
    def add_cell(self):
        pass
    
    def add_output_layer(self):
        pass
    
    def compute_cost(self):
        pass
    
    def ms_error(self):
        pass
    
    def _weight_variable(self):
        pass
    
    def _bias_variable(self):
        pass