# -*- coding: utf-8 -*-

"""
Author:XiyouZhaoC
Rethinking the Inception Architecture for Computer Vision
    InceptionV2
    
YawnDD is a dataset which contains 30 videos to classify wheather a driver is yawning
We manmuly labeled the pictures which have detected face into 10340 training examples
and 1449 testing examples
"""

import tensorflow as tf
import sys
import process_data as pd
import datetime

n_train_example = 10340
n_test_example = 1449

learning_rate = 0.001
dropout_keep_prob = 0.8
batch_size = 100
n_class = 2


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
inception v3
"""        
def inception_v3(x,
                 width,
                 height):
    with tf.name_scope('Image_Reshape'):
        x = tf.reshape(x,[-1,width,height,1])
    print('Image reshape,shape={0}'.format(x.get_shape()))

    # 299 x 299 x 1 => 149 x 149 x 32
    name='Conv2d_1a_3x3'
    with tf.name_scope(name):
        net = conv2d(x,n_filters=32,
                     k_h=3,k_w=3,
                     stride_h=2,stride_w=2,
                     padding='VALID',
                     name=name+'/Conv2d_1a_3x3')
    print('Conv2d_1a_3x3,shape={0}'.format(net.get_shape()))
    
    # 149 x 149 x 32 => 147 x 147 x 32
    name='Conv2d_2a_3x3'
    with tf.name_scope(name):
        net = conv2d(net,n_filters=32,
                     k_h=3,k_w=3,
                     stride_h=1,stride_w=1,
                     padding='VALID',
                     name=name+'/Conv2d_2a_3x3')
    print('Conv2d_2a_3x3,shape={0}'.format(net.get_shape()))
    
    # 147 x 147 x 32 => 147 x 147 x 64
    name='Conv2d_2b_3x3'
    with tf.name_scope(name):
        net = conv2d(net,n_filters=64,
                     k_h=3,k_w=3,
                     stride_h=1,stride_w=1,
                     padding='SAME',
                     name=name+'/Conv2d_2b_3x3')
    print('Conv2d_2b_3x3,shape={0}'.format(net.get_shape()))
    
    # 147 x 147 x 64 => 73 x 73 x 64
    name='MaxPool_3a_3x3'
    with tf.name_scope(name):
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='VALID')
    print('MaxPool_3a_3x3,shape={0}'.format(net.get_shape()))
    
    # 73 x 73 x 64 => 73 x 73 x 80
    name='Conv2d_3b_1x1'
    with tf.name_scope(name):
        net = conv2d(net,n_filters=80,
                     k_h=1,k_w=1,
                     stride_h=1,stride_w=1,
                     padding='VALID',
                     name=name+'/Conv2d_3b_1x1')
    print('Conv2d_3b_1x1,shape={0}'.format(net.get_shape()))
    
    # 73 x 73 x 80 => 71 x 71 x 192
    name='Conv2d_4a_3x3'
    with tf.name_scope(name):
        net = conv2d(net,n_filters=192,
                     k_h=3,k_w=3,
                     stride_h=1,stride_w=1,
                     padding='VALID',
                     name=name+'/Conv2d_4a_3x3')
    print('Conv2d_4a_3x3,shape={0}'.format(net.get_shape()))
    
    # 71 x 71 x 192 => 35 x 35 x 192
    name='MaxPool_5a_3x3'
    with tf.name_scope(name):
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='VALID')
    print('MaxPool_5a_3x3,shape={0}'.format(net.get_shape()))
    
    # =========================Inception Blocks================================
    # 35 x 35 x 192 => 35 x 35 x 256
    name='Mixed_5b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=48,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=64,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_5x5_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_5b,shape={0}'.format(net.get_shape()))
    
    # 35 x 35 x 256 => 35 x 35 x 288
    name='Mixed_5c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=48,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=64,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_5x5_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_5c,shape={0}'.format(net.get_shape()))
    
    # 35 x 35 x 288 => 35 x 35 x 288
    name='Mixed_5d'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=48,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=64,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_5x5_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_5d,shape={0}'.format(net.get_shape()))
    
    # 35 x 35 x 288 => 17 x 17 x 768
    name='Mixed_6a'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=384,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='VALID',
                              name=name+'/Conv2d_1a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
            branch_1 = conv2d(branch_1,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='VALID',
                              name=name+'/Conv2d_1a_1x1_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='VALID')
        net = tf.concat(3, [branch_0, branch_1, branch_2])
    print('Mixed_6a,shape={0}'.format(net.get_shape()))
    
    # 17 x 17 x 768 => 17 x 17 x 768
    name='Mixed_6b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=128,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x7_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_7x1_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_1x7_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0d_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0e_1x7_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])   
    print('Mixed_6b,shape={0}'.format(net.get_shape()))
    
    # 17 x 17 x 768 => 17 x 17 x 768
    name='Mixed_6c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=160,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x7_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_7x1_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_1x7_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0d_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0e_1x7_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])   
    print('Mixed_6c,shape={0}'.format(net.get_shape()))
    
    #  17 x 17 x 768 =>  17 x 17 x 768
    name='Mixed_6d'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=160,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x7_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_7x1_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_1x7_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0d_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0e_1x7_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])   
    print('Mixed_6d,shape={0}'.format(net.get_shape()))
    
    #  17 x 17 x 768 =>  17 x 17 x 768
    name='Mixed_6e'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x7_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_7x1_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_1x7_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0d_7x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0e_1x7_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])   
    print('Mixed_6e,shape={0}'.format(net.get_shape()))
    
    # 17 x 17 x 768 => 8 x 8 x 1280
    name='Mixed_7a'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
            branch_0 = conv2d(branch_0,n_filters=320,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='VALID',
                              name=name+'/Conv2d_1a_3x3_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=1,k_w=7,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x7_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=7,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_7x1_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='VALID',
                              name=name+'/Conv2d_1a_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='VALID')
        net = tf.concat(3, [branch_0, branch_1, branch_2])
    print('Mixed_7a,shape={0}'.format(net.get_shape()))
           
    # 8 x 8 x 1280 => 8 x 8 x 2048
    name='Mixed_7b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=320,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=384,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = tf.concat(3,[
                    conv2d(branch_1,n_filters=384,k_h=1,k_w=3,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0b_1x3_b1'),
                    conv2d(branch_1,n_filters=384,k_h=3,k_w=1,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0c_3x1_b1')])
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=448,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=384,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = tf.concat(3,[
                    conv2d(branch_2,n_filters=384,k_h=1,k_w=3,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0c_1x3_b2'),
                    conv2d(branch_2,n_filters=384,k_h=3,k_w=1,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0d_3x1_b2')])
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_7b,shape={0}'.format(net.get_shape()))
    
    # 8 x 8 x 2048 => 8 x 8 x 2048
    name='Mixed_7c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=320,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=384,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = tf.concat(3,[
                    conv2d(branch_1,n_filters=384,k_h=1,k_w=3,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0b_1x3_b1'),
                    conv2d(branch_1,n_filters=384,k_h=3,k_w=1,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0c_3x1_b1')])
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=448,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=384,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = tf.concat(3,[
                    conv2d(branch_2,n_filters=384,k_h=1,k_w=3,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0c_1x3_b2'),
                    conv2d(branch_2,n_filters=384,k_h=3,k_w=1,
                           stride_h=1,stride_w=1,padding='SAME',name=name+'Conv2d_0d_3x1_b2')])
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_7c,shape={0}'.format(net.get_shape()))
    
    return net
    
def prediction(x,dropout_keep_prob):
    # avg_pool
    with tf.name_scope('Avg_pool'):
        net = tf.nn.avg_pool(x,ksize=[1,
                                      x.get_shape().as_list()[1],
                                      x.get_shape().as_list()[2],
                                      1],
                            strides=[1,1,1,1],
                            padding='VALID')
    print('Average Pool,shape={0}'.format(net.get_shape()))
    
    with tf.name_scope('Dropout'):
        net = tf.nn.dropout(net,dropout_keep_prob)
    
    # Flatten
    with tf.name_scope('Flatten_layer'):
        net = tf.reshape(
            net,
            [-1, net.get_shape().as_list()[1] *
             net.get_shape().as_list()[2] *
             net.get_shape().as_list()[3]])
    print('Flatten,shape={0}'.format(net.get_shape()))
    
    weights = tf.Variable(tf.random_normal([net.get_shape().as_list()[1],n_class]))
    biases = tf.Variable(tf.constant(0.1,shape=[n_class,]))
    
     # softmax
    with tf.name_scope('SoftMax'):
        results = tf.matmul(net,weights)+biases
        results = tf.nn.softmax(results)
    print('Result,shape={0}'.format(results.get_shape()))
    return results    
  
def test_net_shape(width,height):
    x = tf.placeholder(tf.float32,[None,width*height])
    y_inception = inception_v3(x,width,height)
    y_pred = prediction(y_inception,0.8)
   

def train_inception_v3(width=256,height=256):
    
    d_start = datetime.datetime.now()
    
    print('...... loading the dataset ......')
    train_set_x,train_set_y,test_set_x,test_set_y = pd.load_data_set(width,height)
    
    x = tf.placeholder(tf.float32,[None,width*height]) # input
    y = tf.placeholder(tf.float32,[None,n_class])      # label
    keep_prob = tf.placeholder(tf.float32)             # dropout_keep_prob
    
    y_inception = inception_v3(x,width,height)
    y_pred = prediction(y_inception,keep_prob)
    
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred),reduction_indices=1))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    best_acc = 0.
    
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        print('...... initializating varibale ...... ')
        sess.run(init)
        
        n_epochs = 100
        print('...... start to training ......')
        for epoch_i in range(n_epochs):
            # Training 
            train_accuracy = 0.
            for batch_i in range(n_train_example//batch_size):
                
                batch_xs = train_set_x[batch_i*batch_size:(batch_i+1)*batch_size]
                batch_ys = train_set_y[batch_i*batch_size:(batch_i+1)*batch_size]
                _,loss,acc = sess.run([optimizer,cost,accuracy],
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                keep_prob:dropout_keep_prob}
                                                )
                print('epoch:{0},minibatch:{1},cost:{2},train_accuracy:{3}'.format(epoch_i,batch_i,loss,acc))
                train_accuracy += acc

            train_accuracy /= (n_train_example//batch_size)
            print('----epoch:{0},training acc = {1}'.format(epoch_i,train_accuracy))
            
            # Validation
            valid_accuracy = 0.
            for batch_i in range(n_test_example//batch_size):
                batch_xs = test_set_x[batch_i*batch_size:(batch_i+1)*batch_size]
                batch_ys = test_set_y[batch_i*batch_size:(batch_i+1)*batch_size]
                valid_accuracy += sess.run(accuracy,
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                keep_prob:1.0})
            valid_accuracy /= (n_test_example//batch_size)
            print('epoch:{0},train_accuracy:{1},valid_accuracy:{2}'.format(epoch_i,train_accuracy,valid_accuracy))
            if(train_accuracy > best_acc):
                best_acc = train_accuracy
    
    d_end = datetime.datetime.now()
    print('...... training finished ......')
    print('...... best accuracy:{0} ......'.format(best_acc))
    print('...... running time:{0}'.format( (d_end-d_start).seconds))
     
if __name__ == '__main__':
    if len(sys.argv) == 3:
        # python *.py width height
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... training Inception v1:width={0},height={1}'.format(w,h))
        train_inception_v3(w,h)
    elif len(sys.argv) == 4:
        # python *.py width height test_net_shape
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... test Inception v1 net shape:width={0},height={1}'.format(w,h))
        test_net_shape(w,h)
    else:
        #len(sys.argv) == 1 python *.py
        pass