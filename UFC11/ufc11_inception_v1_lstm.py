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

# Dataset count
n_train_example = 33528
n_test_example = 4872

# Network Parameter
learning_rate = 0.001

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
    with tf.name_scope('Input_reshape'):
        # x[pic_batch_size,width*height] ==> [pic_batch_size,width,height,1]
        x = tf.reshape(x,[-1,width,height,1])
        tf.image_summary('input',x,11)# TODO
    with tf.name_scope('Conv2d_1a_7x7'):
        # 7*7 conv stride = 2
        net = conv2d(x,n_filters=64,
                         k_h=7,k_w=7,
                         stride_h=2,stride_w=2,
                         name='Conv2d_1a_7x7',
                         padding='SAME'
                         )
        
    with tf.name_scope('MaxPool_2a_3x3'):
        # MaxPool_2a_3*3 3*3 maxpool stride=2  
        net = tf.nn.max_pool(net,
                             ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME')    
        
    with tf.name_scope('Conv2d_2b_1x1'):           
        # Conv2d_2b_1*1 1*1 conv stride=1
        net = conv2d(net,n_filters=64,
                     k_h=1,k_w=1,
                     stride_h=1,stride_w=1,
                     name='Conv2d_2b_1x1',
                     padding='SAME')
    
    with tf.name_scope('Conv2d_2c_3x3'):
        # Conv2d_2c_3*3 3*3 conv stride =1
        net = conv2d(net,n_filters=192,
                     k_h=3,k_w=3,
                     stride_h=1,stride_w=1,
                     name='Conv2d_2c_3x3',
                     padding='SAME')
    
    with tf.name_scope('MaxPool_3a_3x3'):
        # MaxPool_3a_3*3 3*3 MaxPool stride=2 
        net = tf.nn.max_pool(net,
                             ksize=[1,3,3,1],
                             strides=[1,2,2,1],
                             padding='SAME')          
    
    with tf.name_scope('Mixed_3b'):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=64,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0a_1x1',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=96,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0a_1x1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=128,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0b_3x3',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=16,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0a_1x1',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=32,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0b_5x5',
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
                              name='Conv2d_0b_1x1',
                              padding='SAME')
        
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
        
    # Mixed_3c
    with tf.name_scope('Mixed_3c'):
        with tf.name_scope('Branch_0'):
            # Branch_0 1*1 conv stride=1
            branch_0 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0a_1x1',
                              padding='SAME')
        with tf.name_scope('Branch_1'):
            # Branch_1 1*1 conv stride=1
            branch_1 = conv2d(net,n_filters=128,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0a_1x1',
                              padding='SAME')
            # Branch_1 3*3 conv stride=1
            branch_1 = conv2d(branch_1,n_filters=192,
                              k_h=3,k_w=3,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0b_3x3',
                              padding='SAME')
        with tf.name_scope('Branch_2'):
            # Branch_2 1*1 conv stride=1
            branch_2 = conv2d(net,n_filters=32,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0a_1x1',
                              padding='SAME')
            # Branch_2 5*5 conv stride=1
            branch_2 = conv2d(branch_2,n_filters=96,
                              k_h=5,k_w=5,
                              stride_h=1,stride_w=1,
                              name='Conv2d_0b_3x3',
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
                              name='Conv2d_0b_1x1',
                              padding='SAME')
        
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    
    with tf.name_scope('MaxPool_4a_3x3')        :
        # MaxPool 3*3 stride=2
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='SAME')

    with tf.name_scope('Mixed_4b'):
        with tf.name_scope('Branch_0'):
            #TODO
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

def lstm_layer(x):
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
    
    # x[pic_batch,n_inputs] ==> [video_batch_size,fps,n_inputs]
    x = tf.reshape(x,[-1,fps,n_inputs])
    # x[video_batch_size,fps,n_inputs] ==> [video_batch_size * fps,n_inputs]
    x = tf.reshape(x,[-1,n_inputs])
    # x_in ==> (video_batch_size * fps,n_hidden_units)
    x_in = tf.matmul(x,weights['in']) + biases['in']
    # x_in ==> (video_batch_size,fps,n_hidden_units)
    x_in = tf.reshape(x_in,[-1,fps,n_hidden_units])
    
    # cell
    # forget_bias = 1.0 represents all information can through lstm
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                            forget_bias=1.0,
                                            state_is_tuple=True)
    _init_state = lstm_cell.zero_state(video_batch_size,dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,
                                       x_in,
                                       initial_state=_init_state,
                                       time_major=False
                                       )
    #==================================DUBUG===================================
                                       
    #[10,24,128][video_batch_size,fps,n_hidden_units]
    print('After LSTM layer dynamic run,output shape = {0}'.format(outputs.get_shape()))
    outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
    
    results = tf.matmul(outputs[-1],weights['out']) + biases['out']
    return results
        
 
def train_inception_v1_lstm(width,height):
    x = tf.placeholder(tf.float32,[None,width*height])
    x = tf.reshape(x,[-1,width,height])
    
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