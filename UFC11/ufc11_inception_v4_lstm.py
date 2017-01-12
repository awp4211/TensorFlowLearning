# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import process_data as pd


#Dataset count
n_train_example = 33528
n_test_example = 4872

#Network Parameter
learning_rate = 0.0001
dropout_keep_prob = 0.8

pic_batch_size = 120 
fps = 24
video_batch_size = pic_batch_size / fps
n_classes = 11

# LSTM Parameter
n_hidden_units = 384


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
        
def inception_v4(x,
                 width,
                 height,
                 dropout_keep_prob):
    with tf.name_scope('Image_Reshape'):
        x = tf.reshape(x,[-1,width,height,1])
    print('IMage reshape,shape={0}'.format(x.get_shape()))
    
    # Stem
    # 299 * 299 * 1 ==> 149 * 149 *32
    name='Stem'
    with tf.name_scope(name+'/Conv2d_1a_3*3'):
        net = conv2d(x,n_filters=32,
                         k_h=3,k_w=3,
                         stride_h=2,stride_w=2,
                         padding='VALID',
                         name=name+'/Conv2d_1a_3*3')
    print('Conv2d_1a_3*3,shape={0}'.format(net.get_shape()))
    
    # 149 * 149 *32 ==> 147 * 147 * 32
    with tf.name_scope(name+'/Conv2d_2a_3*3'):
        net = conv2d(net,n_filters=32,
                         k_h=3,k_w=3,
                         stride_h=1,stride_w=1,
                         padding='VALID',
                         name=name+'/Conv2d_2a_3*3')
    print('Stem Conv2d_2a_3*3,shape={0}'.format(net.get_shape()))
    
    # 147 * 147 * 32 ==> 147 * 147 * 64
    with tf.name_scope(name+'/Conv2d_2b_3*3'):
        net = conv2d(net,n_filters=64,
                         k_h=3,k_w=3,
                         stride_h=1,stride_h=1,
                         padding='SAME',
                         name=name+'/Conv2d_2b_3*3')
    print('Stem Conv2d_2b_3*3,shape={0}'.format(net.get_shape()))
    
    # 147 * 147 * 64 ==> 147 * 147 * 160
    name = 'Stem/Mixed_3a'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=96,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_0')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                          strides=[1,2,2,1],
                                          padding='SAME')
        net = tf.concat(3,[branch_0,branch_1])
    print('Stem Mixed_3a,shape={0}'.format(net.get_shape()))
    
    # 147 * 147 * 160 ==> 71 * 71 * 192
    name = 'Stem/Mixed_3b'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=64,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_0/Conv2d_0a_1*1')
            branch_0 = conv2d(branch_0,n_filters=96,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='VALID',
                                  name=name+'/Branch_0/Conv2d_0b_3*3')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=64,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1*1')
            branch_1 = conv2d(branch_1,n_filters=64,
                                  k_h=1,k_w=7,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/conv2d_1b_7*1')
            branch_1 = conv2d(branch_1,n_filters=64,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/conv2d_1b_1*7')
            branch_1 = conv2d(branch_1,n_filters=96,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='VALID',
                                  name=name+'/Branch_1/conv2d_1c_3*3')
        net = tf.concat(3,[branch_0,branch_1])
    print('Stem Mixed_3b,shape={0}'.format(net.get_shape()))

    # 71 * 71 * 192 ==> 35 * 35 * 384
    name = 'Stem/Mixed_3c'
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='VALID',
                                  name=name+'/Branch_0/conv2d_1a_3*3')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                          strides=[1,2,2,1],
                                          padding='VALID')
        net = tf.concat(3,[branch_0,branch_1])
    print('Stem Mixed_3c,shape={0}'.format(net.get_shape()))
    
    # 4 * Inception-A
    # 35 * 35 * 384 ==> 35 * 35 * 384
    name = 'InceptionA_1'
    net = inception_A_block(net,name)
    print('InceptionA_1,shape={0}'.format(net.get_shape()))
    
    # 35 * 35 * 384 ==> 35 * 35 * 384
    name = 'InceptionA_2'
    net = inception_A_block(net,name)
    print('InceptionA_2,shape={0}'.format(net.get_shape()))
    
    # 35 * 35 * 384 ==> 35 * 35 * 384
    name = 'InceptionA_3'
    net = inception_A_block(net,name)
    print('InceptionA_3,shape={0}'.format(net.get_shape()))
    
    # 35 * 35 * 384 ==> 35 * 35 * 384
    name = 'InceptionA_4'
    net = inception_A_block(net,name)
    print('InceptionA_4,shape={0}'.format(net.get_shape()))
    
    # Reduction-A
    # 35 * 35 * 384 ==> 17 * 17 * 1024
    name = 'ReductionA'
    net = reduction_A_block(net,name)
    print('ReductionA,shape={0}'.format(net.get_shape()))

    # 7*InceptionB
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_1'
    net = inception_B_block(net,name)
    print('InceptionB_1,shape={0}'.format(net.get_shape()))
    
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_2'
    net = inception_B_block(net,name)
    print('InceptionB_2,shape={0}'.format(net.get_shape()))
    
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_3'
    net = inception_B_block(net,name)
    print('InceptionB_3,shape={0}'.format(net.get_shape()))
    
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_4'
    net = inception_B_block(net,name)
    print('InceptionB_4,shape={0}'.format(net.get_shape()))
    
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_5'
    net = inception_B_block(net,name)
    print('InceptionB_5,shape={0}'.format(net.get_shape()))
    
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_6'
    net = inception_B_block(net,name)
    print('InceptionB_6,shape={0}'.format(net.get_shape()))
    
    # 17 * 17 * 1024 ==> 17 * 17 * 1024
    name = 'InceptionB_7'
    net = inception_B_block(net,name)
    print('InceptionB_7,shape={0}'.format(net.get_shape()))
    
    # Reduction-B
    name = 'ReductionB'
    net = reduction_B_block(net,name)
    print('ReductionB,shape={0}'.format(net.get_shape()))
    
    
    
def inception_A_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                      strides=[1,1,1,1],
                                      padding='SAME',
                                      name=name+'/Branch_0/Avg_pool')
            branch_0 = conv2d(branch_0,n_filters=96,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_0/Conv2d_0a_1*1')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=96,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1*1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=64,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1*1')
            branch_2 = conv2d(branch_2,n_filters=96,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2b_3*3')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = conv2d(net,n_filters=64,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3a_1*1')
            branch_3 = conv2d(branch_3,n_filters=96,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3b_3*3')
            branch_3 = conv2d(branch_3,n_filters=96,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3c_3*3')
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    return net
            
def reduction_A_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                          strides=[1,2,2,1],
                                          padding='VALID',
                                          name=name+'/Branch_0/Max_pool_0a')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=384,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_1/Conv2d_1a_3*3')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=192,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1*1')
            branch_2 = conv2d(branch_2,n_filters=224,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2b_3*3')
            branch_2 = conv2d(branch_2,n_filters=256,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_2/Conv2d_2c_3*3')
        net = tf.concat(3,[branch_0,branch_1,branch_2])
    return net

def inception_B_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = tf.nn.avg_pool(net,ksize=[1,2,2,1],
                                          strides=[1,1,1,1],
                                          padding='SAME',
                                          name=name+'/Branch_0/Avg_pool_0a')
            branch_0 = conv2d(branch_0,n_filters=128,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_0/Conv2d_0b_1*1')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=384,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1*1')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=192,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1*1')
            branch_2 = conv2d(branch_2,n_filters=224,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2b_1*7')
            branch_2 = conv2d(branch_2,n_filters=256,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2c_1*7')
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = conv2d(net,n_filters=192,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3a_1*1')
            branch_3 = conv2d(branch_3,n_filters=192,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3b_1*7')
            branch_3 = conv2d(branch_3,n_filters=224,
                                  k_h=1,k_w=7,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3c_7*1')
            branch_3 = conv2d(branch_3,n_filters=224,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3c_1*7')
            branch_3 = conv2d(branch_3,n_filters=256,
                                  k_h=1,k_w=7,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3d_7*1')
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    return net
    
def reduction_B_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                          strides=[1,2,2,1],
                                          padding='VALID',
                                          name=name+'/Branch_0/Max_pool_0a')
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=192,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1*1')
            branch_1 = conv2d(branch_1,n_filters=192,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_1/Conv2d_1b_3*3')
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=256,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1*1')
            branch_2 = conv2d(branch_2,n_filters=256,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2b_1*7')
            branch_2 = conv2d(branch_2,n_filters=256,
                                  k_h=1,k_w=7,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2c_7*1')
            branch_2 = conv2d(branch_2,n_filters=320,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_2/Conv2d_2d_3*3')
        net = tf.concat(3,[branch_0,branch_1,branch_2])
    return net