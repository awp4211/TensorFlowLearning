# -*- coding: utf-8 -*-

"""
Author:XiyouZhaoC
Rethinking the Inception Architecture for Computer Vision
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

pic_batch_size = 240 
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
    with tf.name_scope('Image_Reshape'):
        x = tf.reshape(x,[-1,width,height,1])
    print('Image reshape,shape={0}'.format(x.get_shape()))
    
    # 224 x 224 x 1 => 112 x 112 x 64
    # image shape = shape / 2
    with tf.name_scope('Conv2d_1a_7x7'):
        net = conv2d(x,n_filters=64,
                   k_h=7,k_w=7,
                   stride_h=2,stride_w=2,
                   stddev=1.0,
                   padding='SAME',
                   name='Conv2d_1a_7x7')
    print('Conv2d_1a_7x7,shape={0}'.format(net.get_shape()))
        
    # 112 x 112 x 64 => 56 x 56 x 64
    # image shape = shape / 2
    with tf.name_scope('MaxPool_2a_3x3'):
        net = tf.nn.max_pool(net,
                             ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME')    
    print('MaxPool_2a_3x3,shape={0}'.format(net.get_shape()))
        
    # 56 x 56 x 64 => 56 x 56 x 64
    with tf.name_scope('Conv2d_2b_1x1'):
        net = conv2d(net,n_filters=64,
                     k_h=1,k_w=1,
                     stride_h=1,stride_w=1,
                     padding='SAME',
                     name='Conv2d_2b_1x1')
    print('Conv2d_2b_1x1,shape={0}'.format(net.get_shape()))
        
    # 56 x 56 x 64 => 56 x 56 x 192
    with tf.name_scope('Conv2d_2c_3x3'):
        net = conv2d(net,n_filters=192,
                     k_h=3,k_w=3,
                     stride_h=1,stride_w=1,
                     padding='SAME',
                     name='Conv2d_2c_3x3')
    print('Conv2d_2c_3x3,shape={0}'.format(net.get_shape()))
        
    # 56 x 56 x 192 => 28 x 28 x 192
    # image shape = shape / 2
    with tf.name_scope('MaxPool_3a_3x3'):
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME')
    print('MaxPool_3a_3x3,shape={0}'.format(net.get_shape()))
        
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
    print('Mixed_3b,shape={0}'.format(net.get_shape()))
        
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
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    print('Mixed_3c,shape={0}'.format(net.get_shape())) 
        
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
    print('Mixed_4a,shape={0}'.format(net.get_shape()))
        
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
                              stride_h=1,stride_w=1,
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
    print('Mixed_4b,shape={0}'.format(net.get_shape()))
        
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
                              stride_h=1,stride_w=1,
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
    print('Mixed_4c,shape={0}'.format(net.get_shape()))
        
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
                              stride_h=1,stride_w=1,
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
    print('Mixed_4d,shape={0}'.format(net.get_shape()))
        
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
                              stride_h=1,stride_w=1,
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
    print('Mixed_4e,shape={0}'.format(net.get_shape()))
        
    # 14 x 14 x 576 => 7 x 7 x 1024
    name='Mixed_5a'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
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
    print('Mixed_5a,shape={0}'.format(net.get_shape()))
        
    # 7 x 7 x 1024 => 7 x 7 x 1024
    name='Mixed_5b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=352,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
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
            branch_2 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=224,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=224,
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
    print('Mixed_5b,shape={0}'.format(net.get_shape()))
        
    # 7 x 7 x 1024 => 7 x 7 x 1024
    name='Mixed_5c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=352,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b0')
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
            branch_2 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.09,
                              padding='SAME',
                              name=name+'/Conv2d_0a_1x1_b2')
            branch_2 = conv2d(branch_2,n_filters=224,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_3x3_b2')
            branch_2 = conv2d(branch_2,n_filters=224,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              name=name+'/Conv2d_0c_3x3_b2')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            branch_3 = conv2d(branch_3,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              stddev=0.1,
                              padding='SAME',
                              name=name+'/Conv2d_0b_1x1_b3')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_5c,shape={0}'.format(net.get_shape()))
    #================================DEBUG=====================================
    # 224*224*1 ===> 7*7*1024 
    # 192*192*1 ===> 6*6*1024
    # 160*160*1 ===> 5*5*1024
    # 128*128*1 ===> 4*4*1024
    # 96*96*1   ===> 3*3*1024
    #================================DEBUG=====================================
    print('After InceptionV2, shape = {0}'.format(net.get_shape()))
    
    with tf.name_scope('Average_pool'):
        net = tf.nn.avg_pool(net,
                             ksize=[1,net.get_shape().as_list()[1],
                                    net.get_shape().as_list()[1],1],
                             strides=[1,1,1,1],
                             padding='VALID')
    print('After avg pool size = {0}'.format(net.get_shape()))
    
    with tf.name_scope('Flatten_layer'):
        net = tf.reshape(
            net,
            [-1,net.get_shape().as_list()[1] * 
            net.get_shape().as_list()[2] * 
            net.get_shape().as_list()[3]])
            
    # Dropout
    with tf.name_scope('Dropout'):
        net = tf.nn.dropout(net,keep_prob=dropout_keep_prob)
    print('After InceptionV2, shape = {0}'.format(net.get_shape()))
    return net

def stacked_lstm_layer(x,
                       n_lstm):
    print('============================LSTM==================================')
    # x :[pic_batch_size,1024]
    # transpose to [video_batch_size,fps,1024]
    # get input     
    n_inputs = x.get_shape().as_list()[-1]
    print('LSTM Layer n_inputs={0}'.format(n_inputs))    

    # Define weights
    weights = {
        #(n_inpus=1024,n_hidden_units=128)
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
        #(n_hidden_units=128,n_classes=11)
        'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
    }
    biases = {
        #(n_hidden_units=128,)
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
        #(n_classes=11,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
    }
    with tf.name_scope('LSMT_reshape'):
        # x[pic_batch,n_inputs] ==> [video_batch_size,fps,n_inputs]
        x = tf.reshape(x,[-1,fps,n_inputs])
        # x[video_batch_size,fps,n_inputs] ==> [video_batch_size * fps,n_inputs]
        x = tf.reshape(x,[-1,n_inputs])
    print('LSTM_reshape,shape = {0}'.format(x.get_shape()))
    
    with tf.name_scope('LSTM_upscale'):
        # x_in ==> (video_batch_size * fps,n_hidden_units)
        x_in = tf.matmul(x,weights['in']) + biases['in']
        # x_in ==> (video_batch_size,fps,n_hidden_units)
        x_in = tf.reshape(x_in,[-1,fps,n_hidden_units])
    print('LSTM_upscale,shape = {0}'.format(x_in.get_shape()))
    
    with tf.name_scope('LSTM_cell'):
        # cell
        # forget_bias = 1.0 represents all information can through lstm
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                                forget_bias=1.0,
                                                state_is_tuple=False)
        # stacked lstm
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_lstm,
                                                   state_is_tuple=False)
        _init_state = stacked_lstm.zero_state(video_batch_size,dtype=tf.float32)
        
        outputs,states = tf.nn.dynamic_rnn(stacked_lstm,
                                           x_in,
                                           initial_state=_init_state,
                                           time_major=False
                                           )
        #==================================DUBUG===================================
                                           
        #[10,24,128][video_batch_size,fps,n_hidden_units]
        print('After LSTM layer dynamic run,output shape = {0}'.format(outputs.get_shape()))
        outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
    print('LSTM_cell,outputs\' shape = {0}'.format(len(outputs)))
    
    with tf.name_scope('LSTM_downscale'):
        results = tf.matmul(outputs[-1],weights['out']) + biases['out']
    print('LSTM_downscale,shape = {0}'.format(results.get_shape()))
    
    return results    

def test_net(width,height):
    x = tf.placeholder(tf.float32,[None,width*height])
    y = tf.placeholder(tf.float32,[None,n_classes])
    y_inception = inception_v2(x,width,height,0.8)
    y_pred = stacked_lstm_layer(y_inception,7)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
    
def train_inception_v2_stacked_lstm(width,
                                    height,
                                    n_lstm=7,
                                    dropout_keep_prob=0.8,
                                    learning_rate=0.00001,
                                    n_epochs=200):
    print('...... loading the dataset ......')
    train_set_x,train_set_y,test_set_x,test_set_y = pd.load_data_set(width,height)
    
    x = tf.placeholder(tf.float32,[None,width*height]) # input
    y = tf.placeholder(tf.float32,[None,n_classes])    # label
    keep_prob = tf.placeholder(tf.float32)             # dropout_keep_prob
    
    y_inception = inception_v2(x,width,height,keep_prob)
    y_pred = stacked_lstm_layer(y_inception,n_lstm)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    best_acc = 0.
    best_acc_epoch = 0
    
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        print('...... initializating varibale ...... ')
        sess.run(init)
        
        print('...... start to training ......')
        for epoch_i in range(n_epochs):
            # Training 
            train_accuracy = 0.
            for batch_i in range(n_train_example//pic_batch_size):
                
                batch_xs = train_set_x[batch_i*pic_batch_size:(batch_i+1)*pic_batch_size]
                batch_ys = train_set_y[batch_i*video_batch_size:(batch_i+1)*video_batch_size]
                _,loss,acc = sess.run([optimizer, cost, accuracy],
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                keep_prob:dropout_keep_prob}
                                                )
                #print('epoch:{0},minibatch:{1},y_res:{2}'.format(epoch_i,batch_i,yy_res))
                #print('epoch:{0},minibatch:{1},y_pred:{2}'.format(epoch_i,batch_i,yy_pred))
                print('epoch:{0},minibatch:{1},cost:{2},train_accuracy:{3}'.format(epoch_i,batch_i,loss,acc))
                train_accuracy += acc

            train_accuracy /= (n_train_example//pic_batch_size)
            print('----epoch:{0},training acc = {1}'.format(epoch_i,train_accuracy))
            
            # Validation
            valid_accuracy = 0.
            for batch_i in range(n_test_example//pic_batch_size):
                batch_xs = test_set_x[batch_i*pic_batch_size:(batch_i+1)*pic_batch_size]
                batch_ys = test_set_y[batch_i*video_batch_size:(batch_i+1)*video_batch_size]
                valid_accuracy += sess.run(accuracy,
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                keep_prob:1.0})
            valid_accuracy /= (n_test_example//pic_batch_size)
            print('epoch:{0},train_accuracy:{1},valid_accuracy:{2}'.format(epoch_i,train_accuracy,valid_accuracy))
            if(train_accuracy > best_acc):
                best_acc_epoch = epoch_i
                best_acc = train_accuracy
            print('---epoch:{0},current best accuracy = {1} in epoch_{2}'.format(epoch_i,best_acc,best_acc_epoch))
    
    print('...... training finished ......')
    print('...... best accuracy{0} @ epoch_{1}......'.format(best_acc,best_acc_epoch))                        
    
    
    
if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... training inception v1 and lstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        train_inception_v2_stacked_lstm(width=w,height=h)
    
    if len(sys.argv) == 4:
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... testing inception v1 and stacked lstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        test_net(width=w,height=h)