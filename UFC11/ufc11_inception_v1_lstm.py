# -*- coding: utf-8 -*-

"""
Author:XiyouZhaoC
Going Deeper with Convolution
    InceptionV1 and LSTM networks
    

UFC11 dataset contains 1600 videos and hava been classified 11 classes 
"""

import tensorflow as tf
import sys
import numpy as np

slim = tf.contrib.slim
trunc_normal = lambda stdddev:tf.truncated_normal_initializer(0.0,stddev)


"""
Defines the Inception V1 base architecture
Args:
    input---a tensor of size[batch_size,height,width,channels]
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
def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      scope='InceptionV1'
                      ):
    end_points = {}
    with tf.variable_scope(scope,'InceptionV1',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                            stride=1,padding='SAME'):
            # conv2d_1a_7*7
            end_point = 'Conv2d_1a_7x7'
            net = slim.conv2d(inputs,64,[7,7],stride=2,scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:return net,end_points

            # MaxPool_2a_3*3    
            end_point = 'MaxPool_2a_3x3'
            net = slim.max_pool2d(net,[3,3],stride=2,scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:return net,end_points
            
            # Conv2d_2b_1*1
            end_point = 'Conv2d_2b_1x1'
            net = slim.conv2d(net,64,[1,1],scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:return net,end_points
            
            # Conv2d_2c_3*3
            end_point = 'Conv2d_2c_3x3'
            net = slim.conv2d(net,192,[3,3],scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:return net,end_points
            
            # MaxPool_3a_3*3
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net,[3,3],stride=2,scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:return net,end_points
            
            # Mixed_3b
            end_point = 'Mixed_3b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,96,[1,1],scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1,128,[3,3],scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,16,[1,1],scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2,32,[3,3],scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net,[3,3],scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')
                net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point:return net,end_points

            # Mixed_3c
            end_point = 'Mixed_3c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1,192,[3,3],scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,32,[1,1],scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net,[3,3],scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0,branch_1,branch_2,branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points
    
            # MaxPool_4a
            end_point = 'MaxPool_4a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points
            
            end_point = 'Mixed_4c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points
            
            end_point = 'Mixed_4d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4f'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            end_point = 'MaxPool_5a_2x2'
            net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1'):
    # Final pooling and prediction
    with tf.variable_scope(scope, 'InceptionV1', [inputs, num_classes],
                         reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
            net, end_points = inception_v1_base(inputs, scope=scope)
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [7, 7], stride=1, scope='MaxPool_0a_7x7')
                net = slim.dropout(net,dropout_keep_prob, scope='Dropout_0b')
                logits = slim.conv2d(net, num_classes, [1, 1], 
                                     activation_fn=None,
                                     normalizer_fn=None, 
                                     scope='Conv2d_0c_1x1')
            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


def train_inception_v1_lstm(width,height):
    pass

if __name__ == '__main__':
    
    if sys.argv[1]:
        if sys.argv[2]:
            print('...... training inception v1 and blstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
            w = int(sys.argv[1])
            h = int(sys.argv[2])
            train_inception_v1_lstm(width=w,height=h)
    else:      
        print('...... training inception v1 and blstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        train_inception_v1_lstm()