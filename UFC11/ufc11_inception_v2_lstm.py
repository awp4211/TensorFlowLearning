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
    
    # 224 x 224 x 1 => 112 x 112 x 64
    # image shape = shape / 2
    with tf.name_scope('Conv2d_1a_7x7'):
        net = conv2d(x,n_filters=64,
                   k_h=7,k_w=7,
                   stride_h=2,stride_w=2,
                   stddev=1.0,
                   padding='SAME',
                   name='Conv2d_1a_7x7'
                   )
    # 112 x 112 x 64 => 56 x 56 x 64
    # image shape = shape / 2
    with tf.name_scope('MaxPool_2a_3x3'):
        net = tf.nn.max_pool(net,
                             ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME')    
    
    # 56 x 56 x 64 => 56 x 56 x 64
    with tf.name_scope('Conv2d_2b_1x1'):
        net = conv2d(net,n_filters=64,
                     k_h=1,k_w=1,
                     stride_h=1,stride_w=1,
                     padding='SAME',
                     name='Conv2d_2b_1x1')
    
    # 56 x 56 x 64 => 56 x 56 x 192
    with tf.name_scope('Conv2d_2c_3x3'):
        net = conv2d(net,n_filters=192,
                     k_h=1,k_w=1,
                     stride_h=3,stride_w=3,
                     padding='SAME',
                     name='Conv2d_2c_3x3')
    
    # 56 x 56 x 192 => 28 x 28 x 192
    # image shape = shape / 2
    with tf.name_scope('MaxPool_3a_3x3'):
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             name='MaxPool_3a_3x3')
    
        
    # 28 x 28 x 192 => 28 x 28 x 256
    name = 'Mixed_3b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              stddev=0.09,
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=64,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2'
                              )
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2'
                              )
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
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    
    # 28 x 28 x 256 => 28 x 28 x 320
    name='Mixed_3c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
     
    # 28 x 28 x 320 => 14 x 14 x 576
    name='Mixed_4a'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
            branch_0 = conv2d(net,n_filters=160,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='SAME',
                              name=name+'/Conv2d_1a_3x3_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
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
                              padding='SAME',
                              name=name+'/Conv2d_1a_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='SAME')
        net = tf.concat(3,[branch_0,branch_1,branch_2])
    
    # 14 x 14 x 576 => 14 x 14 x 576
    name='Mixed_4b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=224,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=96,
                              k_h=3,k_w=3,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    
    # 14 x 14 x 576 => 14 x 14 x 576
    name= 'Mixed_4c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=128,
                              k_h=3,k_w=3,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    
    # 14 x 14 x 576 => 14 x 14 x 576
    name='Mixed_4d'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=160,
                              k_h=3,k_w=3,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=160,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
        
    # 14 x 14 x 576 => 14 x 14 x 576
    name='Mixed_4e'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=3,k_w=3,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=192,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    
    # 14 x 14 x 576 => 7 x 7 x 1024
    name='Mixed_5a'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_1a_3x3_b0')
            branch_0 = conv2d(branch_0,n_filters=192,
                              k_h=2,k_w=2,
                              stride_h=2,stride_w=2,
                              padding='SAME',
                              name=name+'/Conv2d_1a_3x3_b0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=256,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
            branch_1 = conv2d(branch_1,n_filters=256,
                              k_h=3,k_w=3,
                              stride_h=2,stride_w=2,
                              padding='SAME',
                              name=name+'/Conv2d_1a_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2])
    
    # 7 x 7 x 1024 => 7 x 7 x 1024
    name='Mixed_5b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=352,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b1')
            branch_1 = conv2d(branch_1,n_filters=320,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2])