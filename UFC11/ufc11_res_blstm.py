# -*- coding: utf-8 -*-

"""
Author:XiyouZhaoC
Residual network and BiLSTM
this code running on tensorflow 0.10.0 platform
By using residual network(ResNet38)(CNN) to extract features and
    using Bidirectional LSTM(double layers LSTM) to process sequence data

UFC11 dataset contains 1600 videos and hava been classified 11 classes 
"""
import tensorflow as tf
import process_data as pd
import sys
import numpy as np


from collections import namedtuple
from math import sqrt


# Dataset count
n_train_example = 33528
n_test_example = 4872

# Network Parameter
learning_rate = 0.01

pic_batch_size = 240 # % fps == 0
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

def residual_network(x,
                     activation=tf.nn.relu):
    LayerBlock = namedtuple(
                'LayerBlock',['num_repeats','num_filters','bottleneck_size'])
    blocks = [LayerBlock(3,128,32),
              LayerBlock(3,256,64),
              LayerBlock(3,512,128),
              LayerBlock(3,1024,256)]
    input_shape = x.get_shape().as_list()
    
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim*ndim!=input_shape[1]:
            raise ValueError('input_shape should be square')
        x = tf.reshape(x,[-1,ndim,ndim,1])
        
    print('input layer,shape = {0}'.format(x.get_shape()))#[batch_size,28,28,1]   
    # First convolution expands to 64 channels and downsamples
    net = conv2d(x,64,k_h=7,k_w=7,
                 name='conv1',activation=activation)#padding=SAME,stride=2
    print('conv1,shape = {0}'.format(net.get_shape()))#[batch_size,14,14,64]
    
    # Max Pool 3*3 kernel pool,stride=2,padding=SAME(kernel-1)/2
    net = tf.nn.max_pool(net,[1,3,3,1],
                         strides=[1,2,2,1],padding='SAME')
    print('max pool1,shape = {0}'.format(net.get_shape()))#[batch,7,7,64]
   
    # Setup first chain of resnets 1*1卷积
    net = conv2d(net,blocks[0].num_filters,k_h=1,k_w=1,
                 stride_h=1,stride_w=1,padding='VALID',name='conv2')
    print('conv2,shape = {0}'.format(net.get_shape()))#[batch,7,7,128]
    
    print('Residual Networds:')
    # Loop through all res blocks
    for block_i,block in enumerate(blocks):
        for repeat_i in range(block.num_repeats):
            name = 'block_%d/repeat_%d'%(block_i,repeat_i)
            
            print('{0} start......'.format(name))
            
            conv = conv2d(net,block.bottleneck_size,k_h=1,k_w=1,
                          stride_h=1,stride_w=1,padding='VALID',
                          activation=activation,
                          name=name+'/conv_in')
            print('{0}/conv_in,shape = {1}'.format(name,conv.get_shape()))
            #!*1卷积[batch,7,7,bottleneck_size]
            
            conv = conv2d(conv,block.bottleneck_size,k_h=3,k_w=3,
                          padding='SAME',stride_h=1,stride_w=1,
                          activation=activation,
                          name=name+'/conv_bottlneck')
            print('{0}/conv_bottleneck,shape = {1}'.format(name,conv.get_shape()))
            #3*3卷积[batch,7,7,bottlneck_size](stride=1,padding=SAME卷积之后维度不变)
            
            conv = conv2d(conv,block.num_filters,k_h=1,k_w=1,
                          padding='VALID',stride_h=1,stride_w=1,
                          activation=activation,
                          name=name+'/conv_out')
            print('{0}/conv_out,shape = {1}'.format(name,conv.get_shape()))
            #1*1卷积[batch,7,7,num_filters]
            
            net = conv + net
            print('{0}/merge,shape = {1}'.format(name,net.get_shape))
            #[batch,7,7,num_filters]
            
        try:
            print('===========================================================')
            print('Next Block (Upscale)')#增加维度
            next_block = blocks[block_i+1]
            name_s = 'block_{0}/conv_upscale'.format(block_i)
            net = conv2d(net,next_block.num_filters,k_h=1,k_w=1,
                         padding='SAME',stride_h=1,stride_w=1,bias=False,name=name_s)
            print('{0},shape = {1}'.format(name_s,net.get_shape()))
        except IndexError:
            pass
    
    # Average Pool
    net = tf.nn.avg_pool(net,
                         ksize=[1,net.get_shape().as_list()[1],
                                net.get_shape().as_list()[2],1],
                         strides=[1,1,1,1],padding='VALID')
    print('Average Pool,shape = {0}'.format(net.get_shape()))# 7*7均值采样
    net = tf.reshape(
        net,
        [-1, net.get_shape().as_list()[1] *
         net.get_shape().as_list()[2] *
         net.get_shape().as_list()[3]])
    print('After residual network shape = {0}'.format(net.get_shape()))
    
    return net


def blstm(x,
          _istate_fw,# forward direction cell initial state
          _istate_bw,# backword direction cell initial state
          ):
    print('============================LSTM==================================')
    
    # x :[pic_batch_size,1024]
    # transpose to [video_batch_size,fps,1024]
    # get input     
    n_inputs = x.get_shape().as_list()[-1]
    print('LSTM Layer n_inputs={0}'.format(n_inputs))

    # Define weights
    weights = {
        #(n_inpus=1024,n_hidden_units=128)
        'hidden':tf.Variable(tf.random_normal([n_inputs,2*n_hidden_units])),
        #(n_hidden_units=128,n_classes=11)
        'out':tf.Variable(tf.random_normal([2*n_hidden_units,n_classes]))
    }
    biases = {
        #(n_hidden_units=128,)
        'hidden':tf.Variable(tf.constant(0.1,shape=[2*n_hidden_units,])),
        #(n_classes=11,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
    }
    
    # BiRNN requires to supply sequence length as [batch_size,int64]
    _seq_len = tf.fill([video_batch_size],tf.constant(fps,dtype=tf.int64))
    
    # x[pic_batch,n_inputs] ==> [video_batch_size,fps,n_inputs]
    x = tf.reshape(x,[-1,fps,n_inputs])
    
    # input shape[video_batch_size,fps,n_inputs] ==> [fps,video_batch_size,n_inputs]
    x = tf.transpose(x,[1,0,2])
    
    # x[video_batch_size,fps,n_inputs] ==> [video_batch_size * fps,n_inputs]
    x = tf.reshape(x,[-1,n_inputs])
    
    # Linear Activation
    x = tf.matmul(x,weights['hidden']) + biases['hidden']

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0)
    
    # split data because rnn cell needs a list of inputs for RNN inner loop
    # ==> fps * [video_batch_size,n_hidden_units]
    x = tf.split(0,fps,x)
    
    outputs = tf.nn.bidirectional_rnn(
        lstm_fw_cell,
        lstm_bw_cell,
        x,
        initial_state_fw=_istate_fw,
        initial_state_bw=_istate_bw,
        sequence_length=_seq_len                                  
    )
    
    return tf.matmul(outputs[-1],weights['out']) + biases['out']

def train_res_blstm(width=256,height=256):
    print('...... loading the dataset ......')
    train_set_x,train_set_y,test_set_x,test_set_y = pd.load_data_set(width,height)
    
    print('...... building the model ......')
    x = tf.placeholder(tf.float32,[None,width*height])
    y = tf.placeholder(tf.float32,[None,n_classes])
    # TensorFlow LSTM cell reuqires 2*n_hidden length(state&cell)
    istate_fw = tf.placeholder(tf.float32,[None,2*n_hidden_units])
    istate_bw = tf.placeholder(tf.float32,[None,2*n_hidden_units])
    
    y_res = residual_network(x)
    y_pred = blstm(y_res,istate_fw,istate_bw)
    
    # Define loss and training functions
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Monitor Accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    best_acc = 0.
    
    init = tf.initialize_all_variables()
    # Session
    with tf.Session() as sess:
        print('...... initializating varibale ...... ')
        sess.run(init)
        
        n_epochs = 100
        print('...... start to training ......')
        for epoch_i in range(n_epochs):
            # Training 
            train_accuracy = 0.
            for batch_i in range(n_train_example//pic_batch_size):
                
                batch_xs = train_set_x[batch_i*pic_batch_size:(batch_i+1)*pic_batch_size]
                batch_ys = train_set_y[batch_i*video_batch_size:(batch_i+1)*video_batch_size]
                _,loss,acc,yy = sess.run([optimizer,cost,accuracy,y_pred],
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                istate_fw:np.zeros((video_batch_size,2*n_hidden_units)),
                                                istate_bw:np.zeros((video_batch_size,2*n_hidden_units))
                                                })
                print('epoch:{0},minibatch:{1},cost:{2},train_accuracy:{3}'.format(epoch_i,batch_i,loss,acc))
                train_accuracy += acc

            train_accuracy /= (n_train_example//pic_batch_size)
            print('epoch:{0},training acc = {1}'.format(epoch_i,train_accuracy))
            
            # Validation
            valid_accuracy = 0.
            for batch_i in range(n_test_example//pic_batch_size):
                batch_xs = test_set_x[batch_i*pic_batch_size:(batch_i+1)*pic_batch_size]
                batch_ys = test_set_y[batch_i*video_batch_size:(batch_i+1)*video_batch_size]
                valid_accuracy += sess.run(accuracy,
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                istate_fw:np.zeros((video_batch_size,2*n_hidden_units)),
                                                istate_bw:np.zeros((video_batch_size,2*n_hidden_units))
                                                })
            valid_accuracy /= (n_test_example//pic_batch_size)
            print('====epoch:{0},train_accuracy:{1},valid_accuracy:{2}'.format(epoch_i,train_accuracy,valid_accuracy))
            if(train_accuracy > best_acc):
                best_acc = train_accuracy
    
    print('...... training finished ......')
    print('...... best accuracy{0} ......'.format(best_acc))


if __name__ == '__main__':
    
    if sys.argv[1]:
        if sys.argv[2]:
            print('...... training res and blstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
            w = int(sys.argv[1])
            h = int(sys.argv[2])
            train_res_blstm(width=w,height=h)
    else:      
        print('...... training res and blstm network:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        train_res_blstm()
   