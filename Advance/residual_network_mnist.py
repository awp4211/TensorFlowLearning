# -*- coding: utf-8 -*-
import tensorflow as tf


"""
Fully connected network
x:input--tensor to network
n_units:int--number of units to connect to 
scope:string--variable scope to use
stddev:float--Initialization's standard deviation
activation:arguments:function while applies a nonlineaeroty

return x:tensor fully-connected output
"""
def linear(x,n_units,scope=None,stddev=0.02,activation=lambda x:x):
    shape = x.get_shape().as_list()
    
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix',[shape[1],n_units],
                                 dtype=tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x,matrix))

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
def conv2d(x,n_filters,
           k_h=5,k_w=5,
           stride_h=2,stride_w=2,
           stddev=0.02,
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
