# -*- coding: utf-8 -*-

"""
Author:XiyouZhaoC
Going Deeper with Convolution
    InceptionV1 and LSTM networks
    

UFC11 dataset contains 1600 videos and hava been classified 11 classes 
"""
import tensorflow as tf
import sys
import process_data as pd

# Dataset count
n_train_example = 33528
n_test_example = 4872

# Network Parameter
learning_rate = 0.001
dropout_keep_prob = 0.8

pic_batch_size = 2400 # % fps == 0
fps = 24
video_batch_size = pic_batch_size / fps
n_classes = 11


# LSTM Parameter
n_hidden_units = 384


"""
2D COnvolution with options for kernel size,stride and init deviation
x:tensor--input tensor to convolution
n_filters:int--Number of filters to apply
k_h,k_w:int--kernel height/width
stride_h,stride_w:int---Stride in rows/cols
stddev:float--Initialization's standard deviation
activation:arguments--Function which applies a nonlinearity
padding:str--'SAME' or 'VALID'
"""        
def conv2d(x,
           n_filters,
           k_h=5,k_w=5,
           stride_h=2,stride_w=2,
           stddev=0.1,
           activation=None,
           bias=True,
           padding='SAME',
           name='Conv2D'):
    with tf.variable_scope(name):
        w = tf.get_variable(
                'w',[k_h,k_w,x.get_shape()[-1],n_filters],
                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x,w,
                            strides=[1,stride_h,stride_w,1],
                            padding=padding)
        if bias:
            b = tf.get_variable('b',
                [n_filters],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.bias_add(conv,b)
        if activation:
            conv = activation(conv)
        return conv  

"""
Inception V2
"""
def inception_v2(x,
                 width,
                 height,
                 dropout_keep_prob):
    with tf.name_scope('Image Reshape'):
        x = tf.reshape(x,[-1,width,height],1)
    
    with tf.name_scope('Conv2d_1a_7x7'):
        x = conv2d(x,n_filters=64,
                   k_h=7,k_w=7,
                   stride_h=2,stride_w=2,
                   stddev=1.0,
                   padding='SAME',
                   name='Conv2d_1a_7x7'
                   )
        
        

