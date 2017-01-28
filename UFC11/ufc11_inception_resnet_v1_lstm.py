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
           activation=tf.nn.relu,
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
        
def inception_resnet_v1(x,
                        width,
                        height,
                        dropout_keep_prob):
    print('=====================Inception ResNet V1==========================')

    # 299 * 299 * 1
    with tf.name_scope('Image_Reshape'):
        x = tf.reshape(x,[-1,width,height,1])
    print('Image reshape,shape={0}'.format(x.get_shape()))
    
    # Stem
    # 299 * 299 * 1 ==> 35 * 35 * 256
    net = stem_block(x)
    
    # 5 * Inception_Resnet_A
    name='Inception_Resnet_A1'
    net = inception_resnet_A_block(net,name)
    
    name='Inception_Resnet_A2'
    net = inception_resnet_A_block(net,name)
    
    name='Inception_Resnet_A3'
    net = inception_resnet_A_block(net,name)
    
    name='Inception_Resnet_A4'
    net = inception_resnet_A_block(net,name)
    
    name='Inception_Resnet_A5'
    net = inception_resnet_A_block(net,name)
    
    # ReductionA
    name='ReductionA'
    net = reduction_A_block(net,name)
    
    # 10 * Inception_Resnet_B
    name = 'Inception_Resnet_B1'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B2'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B3'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B4'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B5'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B6'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B7'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B8'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B9'
    net = inception_resnet_B_block(net,name)
    
    name = 'Inception_Resnet_B10'
    net = inception_resnet_B_block(net,name)

    # ReductionB
    name = 'ReductionB'
    net = reduction_B_block(net,name)
    
    # 5 * Inception_Resnet_C
    name = 'Inception_Resnet_C_1'
    net = inception_resnet_C_block(net,name)
    
    name = 'Inception_Resnet_C_2'
    net = inception_resnet_C_block(net,name)

    name = 'Inception_Resnet_C_3'
    net = inception_resnet_C_block(net,name)

    name = 'Inception_Resnet_C_4'
    net = inception_resnet_C_block(net,name)

    name = 'Inception_Resnet_C_5'
    net = inception_resnet_C_block(net,name)
    
    # AVGPOOL
    with tf.name_scope('Average_pool'):
        net = tf.nn.avg_pool(net,ksize=[1,net.get_shape().as_list()[1],
                                          net.get_shape().as_list()[1],1],
                             strides=[1,1,1,1],
                             padding='VALID')
    print('After avg pool,shape={0}'.format(net.get_shape()))
    
    # FALTTEN
    with tf.name_scope('Flatten_layer'):
        net = tf.reshape(net,
                         [-1,net.get_shape().as_list()[1] * 
                             net.get_shape().as_list()[2] *
                             net.get_shape().as_list()[3]])
    print('After flatten layer,shape={0}'.format(net.get_shape()))
    
    with tf.name_scope('Dropout'):
        net = tf.nn.dropout(net,keep_prob=dropout_keep_prob)
    
    # Final pooling
    print('After pure InceptionV4,shape={0}'.format(net.get_shape()))
    return net
    

def stem_block(net):
    name='Stem'
    with tf.name_scope(name):
        # 299 * 299 * 1 ==> 149 * 149 * 32
        with tf.name_scope(name+'/Conv2d_1a_3x3'):
            net = conv2d(net,n_filters=32,
                             k_h=3,k_w=3,
                             stride_h=2,stride_w=2,
                             padding='VALID',
                             name=name+'/Conv2d_1a_3x3')
            print(name+'/Conv2d_1a_3x3,shape={0}'.format(net.get_shape()))
            
        # 149 * 149 * 32 ==> 147 * 147 * 32
        with tf.name_scope(name+'/Conv2d_1b_3x3'):
            net = conv2d(net,n_filters=32,
                             k_h=2,k_w=2,
                             stride_h=1,stride_w=1,
                             padding='VALID',
                             name=name+'/Conv2d_1b_3x3')
            print(name+'/Conv2d_1b_3x3,shape={0}'.format(net.get_shape()))
        
        # 147 * 147 * 32 ==> 147 * 147 * 64
        with tf.name_scope(name+'/Conv2d_1c_3x3'):
            net = conv2d(net,n_filters=64,
                             k_h=3,k_w=3,
                             stride_h=1,stride_w=1,
                             padding='SAME',
                             name=name+'/Conv2d_1c_3x3')
            print(name+'/Conv2d_1c_3x3,shape={0}'.format(net.get_shape()))
        
        # 147 * 147 * 64 ==> 73 * 73 * 64
        with tf.name_scope(name+'/Maxpool_1d_3x3'):
            net = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                     strides=[1,2,2,1],
                                     padding='VALID',
                                     name=name+'/Maxpool_1d_3x3')
            print(name+'/Maxpool_3x3,shape={0}'.format(net.get_shape()))
            
        # 73 * 73 * 64 ==> 73 * 73 * 80
        with tf.name_scope(name+'/Conv2d_1e_1x1'):
            net = conv2d(net,n_filters=80,
                             k_h=1,k_w=1,
                             stride_h=1,stride_w=1,
                             padding='SAME',
                             name=name+'/Conv2d_1e_1x1')
            print(name+'/Conv2d_1e_1x1,shape={0}'.format(net.get_shape()))
            
        # 73 * 73 * 80 ==> 71 * 71 * 192
        with tf.name_scope(name+'/Conv2d_1f_3x3'):
            net = conv2d(net,n_filters=192,
                             k_h=3,k_w=3,
                             stride_h=1,stride_w=1,
                             padding='VALID',
                             name=name+'/Conv2d_1f_3x3')
            print(name+'/Conv2d_1f_3x3,shape={0}'.format(net.get_shape()))
            
        # 71 * 71 * 192 ==> 35 * 35 * 256
        with tf.name_scope(name+'/Conv2d_1g_3x3'):
            net = conv2d(net,n_filters=256,
                             k_h=3,k_w=3,
                             stride_h=2,stride_w=2,
                             padding='VALID',
                             name=name+'/Conv2d_1g_3x3')
            print(name+'/Conv2d_1g_3x3,shape={0}'.format(net.get_shape()))
    print('Stem,shape={0}'.format(net.get_shape()))
    return net
           
def inception_resnet_A_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/ReLU1'):
            net = tf.nn.relu(net)
        
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=32,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_0/Conv2d_0a_1x1')
            print(name+'/Branch_0/Conv2d_0a_1x1,shape={0}'.format(branch_0.get_shape()))
            
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=32,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1x1')
            branch_1 = conv2d(branch_1,n_filters=32,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1b_3x3')
            print(name+'/Branch_1,shape={0}'.format(branch_1.get_shape()))
        
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=32,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1x1')
            branch_2 = conv2d(branch_2,n_filters=32,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2b_3x3')
            branch_2 = conv2d(branch_2,n_filters=32,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2c_3x3')
            print(name+'/Branch_2,shape={0}'.format(branch_2.get_shape()))
        
        with tf.name_scope(name+'/Merge'):
            conv = tf.concat(3,[branch_0,branch_1,branch_2]) 
            print(name+'/Merge,shape={0}'.format(conv.get_shape()))
        
        with tf.name_scope(name+'/Merge_Conv_1x1'):
            conv = conv2d(conv,n_filters=256,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              activation=None,
                              name=name+'/Merge_Conv_1x1')
            print(name+'/Merge_Conv_1x1,shape={0}'.format(conv.get_shape()))

        with tf.name_scope(name+'/Residual_Merge'):
            net = conv + net
            print(name+'/Residual_Merge,shape={0}'.format(net.get_shape()))
    
        with tf.name_scope(name+'/ReLU2'):
            net = tf.nn.relu(net)
    print(name+',shape={0}'.format(net.get_shape()))        
    return net
            

def reduction_A_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = tf.nn.avg_pool(net,ksize=[1,3,3,1],
                                          strides=[1,2,2,1],
                                          padding='VALID',
                                          name=name+'/Branch_0/Max_pool_0a')
            print(name+'/Branch_0,shape={0}'.format(branch_0.get_shape()))
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=384,#n
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_1/Conv2d_1a_3x3')
            print(name+'/Branch_1,shape={0}'.format(branch_1.get_shape()))
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=192,#k
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1x1')
            branch_2 = conv2d(branch_2,n_filters=192,#l
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2b_3x3')
            branch_2 = conv2d(branch_2,n_filters=256,#m
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_2/Conv2d_2c_3x3')
            print(name+'/Branch_2,shape={0}'.format(branch_2.get_shape()))
        net = tf.concat(3,[branch_0,branch_1,branch_2])
    print(name+' shape={0}'.format(net.get_shape()))
    return net
            
def inception_resnet_B_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/ReLU1'):
            net = tf.nn.relu(net)
        
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=128,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_0/Conv2d_0a_1x1')
            print(name+'/Branch_0,shape={0}'.format(branch_0.get_shape()))
        
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=128,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1x1')
            branch_1 = conv2d(branch_1,n_filters=128,
                                  k_h=7,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1b_1x7')
            branch_1 = conv2d(branch_1,n_filters=128,
                                  k_h=1,k_w=7,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1b_7x1')
            print(name+'/Branch_1,shape={0}'.format(branch_1.get_shape()))

        with tf.name_scope(name+'/Merge'):
            conv = tf.concat(3,[branch_0,branch_1])
            print(name+'/Merge,shape={0}'.format(conv.get_shape()))
            
        with tf.name_scope(name+'/Merge_Conv_1x1'):
            conv = conv2d(conv,n_filters=896,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              activation=None,
                              name=name+'/Merge_Conv_1x1')
            print(name+'/Merge_Conv_1x1,shape={0}'.format(conv.get_shape()))
        
        with tf.name_scope(name+'/Residual_Merge'):
            net = conv + net
            print(name+'/Residual_Merge,shape={0}'.format(net.get_shape()))
    
        with tf.name_scope(name+'/ReLU2'):
            net = tf.nn.relu(net)
    print(name+',shape={0}'.format(net.get_shape()))        
    return net
        
def reduction_B_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = tf.nn.max_pool(net,ksize=[1,3,3,1],
                                          strides=[1,2,2,1],
                                          padding='VALID',
                                          name=name+'/Branch_0/Max_pool_0a')
            print(name+'/Branch_0,shape={0}'.format(branch_0.get_shape()))
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=256,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1x1')
            branch_1 = conv2d(branch_1,n_filters=384,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_1/Conv2d_1b_3x3')
            print(name+'/Branch_1,shape={0}'.format(branch_1.get_shape()))
        with tf.name_scope(name+'/Branch_2'):
            branch_2 = conv2d(net,n_filters=256,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_2/Conv2d_2a_1x1')
            branch_2 = conv2d(branch_2,n_filters=256,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_2/Conv2d_2b_3x3')
            print(name+'/Branch_2,shape={0}'.format(branch_2.get_shape()))
        
        with tf.name_scope(name+'/Branch_3'):
            branch_3 = conv2d(net,n_filters=256,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3a_1x1')
            branch_3 = conv2d(branch_3,n_filters=256,
                                  k_h=3,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_3/Conv2d_3b_3x3')
            branch_3 = conv2d(branch_3,n_filters=256,
                                  k_h=3,k_w=3,
                                  stride_h=2,stride_w=2,
                                  padding='VALID',
                                  name=name+'/Branch_3/Conv2d_3c_3x3')
            print(name+'/Branch_3,shape={0}'.format(branch_3.get_shape()))
        
        with tf.name_scope(name+'/Merge'):
            net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
    print(name+' shape={0}'.format(net.get_shape()))
    return net


def inception_resnet_C_block(net,name):
    with tf.name_scope(name):
        with tf.name_scope(name+'/ReLU1'):
            net = tf.nn.relu(net)
        
        with tf.name_scope(name+'/Branch_0'):
            branch_0 = conv2d(net,n_filters=192,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_0/Conv2d_0a_1x1')
            print(name+'/Branch_0,shape={0}'.format(branch_0.get_shape()))
        
        with tf.name_scope(name+'/Branch_1'):
            branch_1 = conv2d(net,n_filters=192,
                                  k_h=1,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1a_1x1')
            branch_1 = conv2d(branch_1,n_filters=192,
                                  k_h=3,k_w=1,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1b_1x3')
            branch_1 = conv2d(branch_1,n_filters=192,
                                  k_h=1,k_w=3,
                                  stride_h=1,stride_w=1,
                                  padding='SAME',
                                  name=name+'/Branch_1/Conv2d_1b_3x1')
            print(name+'/Branch_1,shape={0}'.format(branch_1.get_shape()))

        with tf.name_scope(name+'/Merge'):
            conv = tf.concat(3,[branch_0,branch_1])
            print(name+'/Merge,shape={0}'.format(conv.get_shape()))
            
        with tf.name_scope(name+'/Merge_Conv_1x1'):
            conv = conv2d(conv,n_filters=1792,
                              k_h=1,k_w=1,
                              stride_h=1,stride_w=1,
                              padding='SAME',
                              activation=None,
                              name=name+'/Merge_Conv_1x1')
            print(name+'/Merge_Conv_1x1,shape={0}'.format(conv.get_shape()))
        
        with tf.name_scope(name+'/Residual_Merge'):
            net = conv + net
            print(name+'/Residual_Merge,shape={0}'.format(net.get_shape()))
    
        with tf.name_scope(name+'/ReLU2'):
            net = tf.nn.relu(net)
    print(name+',shape={0}'.format(net.get_shape()))        
    return net
            
def stacked_lstm_layer(x,
                       n_lstm):
    print('===========================LSTM===================================')
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
    y_inception = inception_resnet_v1(x,width,height,dropout_keep_prob)
    y_pred = stacked_lstm_layer(y_inception,7)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

def train_inception_resnet_v1_stacked_lstm(width,
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
    
    y_inception = inception_resnet_v1(x,width,height,keep_prob)
    y_pred = stacked_lstm_layer(y_inception,n_lstm)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
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
            print('========epoch:{0},current best accuracy = {1} in epoch_{2}'.format(epoch_i,best_acc,best_acc_epoch))
    
    print('...... training finished ......')
    print('...... best accuracy{0} @ epoch_{1}......'.format(best_acc,best_acc_epoch))                 

if __name__ == '__main__':
    if len(sys.argv) == 3:
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... training inception_resnet_v1 and stacked lstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        train_inception_resnet_v1_stacked_lstm(width=w,height=h)

    if len(sys.argv) == 4:
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... testing inception_resnet_v1 and stacked lstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        test_net(width=w,height=h)
    