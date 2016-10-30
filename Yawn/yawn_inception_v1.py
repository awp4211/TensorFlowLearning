# -*- coding: utf-8 -*-
"""
Author:XiyouZhaoC
Going Deeper with Convolution
    InceptionV1
    
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

learning_rate = 0.01
dropout_keep_prob = 0.8
batch_size = 200
n_class = 2
hidden_units = [512,128]

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
Defines the Inception V1 base architecture
Args:
    input---a tensor of size[batch_size,height*width]
    final_endpoint---specifies the endpoint to construct the network up to.
        It can be one of
        ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.
Returns:
    A dictionary from components of the network to the corresponding activation
    
"""
def inception_v1(x,
                 width,
                 height,
                 ):
    # x[batch_size,width*height] => [batch_size,width,height,1]
    with tf.name_scope('Input_reshape'):
        x = tf.reshape(x,[-1,width,height,1])
        tf.image_summary('input',x,2)
    print('Image reshape,shape={0}'.format(x.get_shape()))
    
    with tf.name_scope('Conv2d_1a_7x7'):
        net = conv2d(x,n_filters=64,
                         k_h=7,k_w=7,
                         stride_h=2,stride_w=2,
                         name='Conv2d_1a_7x7',
                         padding='SAME'
                         )
    print('Conv2d_1a_7x7,shape={0}'.format(net.get_shape()))
        
    with tf.name_scope('MaxPool_2a_3x3'):
        # MaxPool_2a_3*3 3*3 maxpool stride=2  
        net = tf.nn.max_pool(net,
                             ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME') 
    print('MaxPool_2a_3x3,shape={0}'.format(net.get_shape()))
    
    with tf.name_scope('LRN_1'):
        net = tf.nn.lrn(net)
    
    
    with tf.name_scope('Conv2d_2b_1x1'):           
        # Conv2d_2b_1*1 1*1 conv stride=1
        net = conv2d(net,n_filters=64,
                     k_h=1,k_w=1,
                     stride_h=1,stride_w=1,
                     name='Conv2d_2b_1x1',
                     padding='SAME')
    print('Conv2d_2b_1x1,shape={0}'.format(net.get_shape()))
    
    with tf.name_scope('Conv2d_2c_3x3'):
        # Conv2d_2c_3*3 3*3 conv stride =1
        net = conv2d(net,n_filters=192,
                     k_h=3,k_w=3,
                     stride_h=1,stride_w=1,
                     name='Conv2d_2c_3x3',
                     padding='SAME')
    print('Conv2d_2c_3x3,shape={0}'.format(net.get_shape()))
    
    with tf.name_scope('MaxPool_3a_3x3'):
        # MaxPool_3a_3*3 3*3 MaxPool stride=2 
        net = tf.nn.max_pool(net,
                             ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME')
    print('MaxPool_3a_3x3,shape={0}'.format(net.get_shape()))          
    
    
    with tf.name_scope('LRN_2'):
        net = tf.nn.lrn(net)
    
    name = 'Mixed_3b'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=128,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=16,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=32,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME')
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    print('Mixed_3b,shape={0}'.format(net.get_shape())) 
        
    # Mixed_3c
    name = 'Mixed_3c'
    with tf.name_scope('Mixed_3c'):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b2',
                              padding='SAME')
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    print('Mixed_3c,shape={0}'.format(net.get_shape())) 
        
    with tf.name_scope('MaxPool_4a_3x3'):
        # MaxPool 3*3 stride=2
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='SAME')
    print('MaxPool_4a_3x3,shape={0}'.format(net.get_shape()))     
    
    name = 'Mixed_4b'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=208,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=16,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=48,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_4b,shape={0}'.format(net.get_shape()))     
    
    name = 'Mixed_4c'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=112,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=224,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=24,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=64,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_4c,shape={0}'.format(net.get_shape()))     
        
    name = 'Mixed_4d'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=256,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=24,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=64,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_4d,shape={0}'.format(net.get_shape()))     
        
    name = 'Mixed_4e'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=112,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=144,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=288,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=64,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_4e,shape={0}'.format(net.get_shape()))     
        
    name = 'Mixed_4f'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=256,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=320,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])

    with tf.name_scope('MaxPool_5a_2x2'):
        net = tf.nn.max_pool(net,ksize=[1,2,2,1],#TODO
                                      strides=[1,2,2,1],
                                      padding='SAME')
    print('Mixed_4f,shape={0}'.format(net.get_shape()))     
        
    name = 'Mixed_5b'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=256,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=160,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=320,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_5b,shape={0}'.format(net.get_shape())) 
    
    name = 'Mixed_5c'
    with tf.name_scope(name):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=384,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b0',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=192,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=384,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_3x3_b1',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=48,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0a_1x1_b2',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=128,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_5x5_b2',
                              padding='SAME') 
        with tf.name_scope('Branch_3'):
            # Branch_3 3*3 maxpool stride=1
            branch_3 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            # Branch_3 1*1 conv stride=1
            branch_3 = conv2d(branch_3,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name=name+'/Conv2d_0b_1x1_b3',
                              padding='SAME')
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
    print('Mixed_5c,shape={0}'.format(net.get_shape())) 
    
    #===========================DEBUG=========================================
    # input width = 224,height = 224 ===> net.shape = [batch,7,7,1024]
    # input width = 192,height = 192 ===> net.shape = [batch,6,6,1024]
    # input witdh = 160,height = 160 ===> net.shape = [batch,5,5,1024]
    #===========================DEBUG=========================================
        
    print('After Inception size = {0}'.format(net.get_shape())) 
    return net       

def pool_and_flatten(x,dropout_keep_prob):
    # avg_pool
    with tf.name_scope('Average_pool'):
        net = tf.nn.avg_pool(x,
                             ksize=[1,x.get_shape().as_list()[1],
                                      x.get_shape().as_list()[1],1],
                             strides=[1,1,1,1],
                             padding='VALID')
    print('After avg pool size = {0}'.format(net.get_shape()))
    
    # Dropout
    with tf.name_scope('Dropout'):
        net = tf.nn.dropout(net,keep_prob=dropout_keep_prob)
    print('Dropout,shape={0}'.format(net.get_shape()))
    
    # Flatten
    with tf.name_scope('Flatten_layer'):
        net = tf.reshape(
            net,
            [-1, net.get_shape().as_list()[1] *
             net.get_shape().as_list()[2] *
             net.get_shape().as_list()[3]])
    print('Flatten,shape={0}'.format(net.get_shape()))
    return net
    
def mlp(net,hidden_units=[512,256,128,64]):
    n_input_size = net.get_shape().as_list()[1]
    weights = {}
    biases = {}
    
    current_input_size = n_input_size
    for i in range(len(hidden_units)):
        w = tf.Variable(tf.random_normal([current_input_size,hidden_units[i]]))
        b = tf.Variable(tf.constant(0.1,shape=[hidden_units[i],]))
        weights.setdefault('w_{0}'.format(i),w)
        biases.setdefault('b_{0}'.format(i),b)
        current_input_size = hidden_units[i]

    # compute
    with tf.name_scope('MLP'):
        current_input = net
        for i in range(len(hidden_units)):
            w = weights.get('w_{0}'.format(i))
            b = biases.get('b_{0}'.format(i))
            net = tf.add(tf.matmul(current_input,w),b)
            net = tf.nn.relu(net)
            current_input = net
            print('MLP Layer {0},shape={1}'.format(i,net.get_shape()))
    

    W = tf.Variable(tf.random_normal([net.get_shape().as_list()[1],n_class]))
    bias = tf.Variable(tf.constant(0.1,shape=[n_class,]))        
    
    # softmax
    with tf.name_scope('SoftMax'):
        results = tf.matmul(net,W)+bias
        results = tf.nn.softmax(results)
    print('Result,shape={0}'.format(results.get_shape()))
    return results
    

def test_net_shape(width,height):
    x = tf.placeholder(tf.float32,[None,width*height])
    y_inception = inception_v1(x,width,height)
    y_pool = pool_and_flatten(y_inception,0.8)
    y_pred = mlp(y_pool,hidden_units=[512,256,128,64])
    
def train_inception_v1(width=256,height=256):
    
    d_start = datetime.datetime.now()
    
    print('...... loading the dataset ......')
    train_set_x,train_set_y,test_set_x,test_set_y = pd.load_data_set(width,height)
    
    x = tf.placeholder(tf.float32,[None,width*height]) # input
    y = tf.placeholder(tf.float32,[None,n_class])    # label
    keep_prob = tf.placeholder(tf.float32)             # dropout_keep_prob
    
    y_inception = inception_v1(x,width,height)
    y_pool = pool_and_flatten(y_inception,keep_prob)
    y_pred = mlp(y_pool,hidden_units=hidden_units)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y))
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
                _,loss,acc,yy_pred = sess.run([optimizer,cost,accuracy,y_pred],
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                keep_prob:dropout_keep_prob}
                                                )
                #print('epoch:{0},minibatch:{1},y_pred:{2}'.format(epoch_i,batch_i,yy_pred))
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
        
        print('=============== Parameters Setting ============================')
        print('learning_rate = {0}'.format(learning_rate))
        print('batch_size = {0}'.format(batch_size))
        print('dropout_keep_prob = {0}'.format(dropout_keep_prob))
        print('hidden_units = {0}'.format(hidden_units))
        print('===============================================================')

        train_inception_v1(w,h)
    elif len(sys.argv) == 4:
        # python *.py width height test_net_shape
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... test Inception v1 net shape:width={0},height={1}'.format(w,h))
        test_net_shape(w,h)
    else:
        #len(sys.argv) == 1 python *.py
        pass