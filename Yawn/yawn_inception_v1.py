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

n_train_example = 10340
n_test_example = 1449

learning_rate = 0.001
dropout_keep_prob = 0.8
batch_size = 100
n_class = 2

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
                 dropout_keep_prob
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

def test_net_shape(width,height):
    x = tf.placeholder(tf.float32,[None,width*height])
    y_inception = inception_v1(x,width,height,0.8)
    
if __name__ == '__main__':
    if len(sys.argv) == 3:
        # python *.py width height
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... training Inception v1:width={0},height={1}'.format(w,h))
    elif len(sys.argv) == 4:
        # python *.py width height test_net_shape
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... test Inception v1 net shape:width={0},height={1}'.format(w,h))
        test_net_shape(w,h)
    else:
        #len(sys.argv) == 1 python *.py
        pass