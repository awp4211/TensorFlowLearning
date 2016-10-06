# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from collections import namedtuple
from math import sqrt

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

"""
Fully connected network
x:input--tensor to network
n_units:int--number of units to connect to 
scope:string--variable scope to use
stddev:float--Initialization's standard deviation
activation:arguments:function while applies a nonlineaeroty

return x:tensor fully-connected output
"""
def linear(x,n_units,scope=None,stddev=0.02,activation=lambda x:x):
    shape = x.get_shape().as_list()
    
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x,matrix))

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
def conv2d(x,n_filters,
           k_h=5,k_w=5,
           stride_h=2,stride_w=2,
           stddev=0.02,
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
Build a residual network
x:placeholder---input to network
n_output:int---Number of outputs of final softmax
activation:arguements---Nonlinearity to apply after each convolution
"""
def residual_network(x,
                     n_output,
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
        
    # First convolution expands to 64 channels and downsamples
    net = conv2d(x,64,k_h=7,k_w=7,
                 name='conv1',activation=activation)
    # Max Pool
    net = tf.nn.max_pool(net,[1,3,3,1],
                         strides=[1,2,2,1],padding='SAME')
    
    # Setup first chain of resnets
    net = conv2d(net,blocks[0].num_filters,k_h=1,k_w=1,
                 stride_h=1,stride_w=1,padding='VALID',name='conv2')
    
    # Loop through all res blocks
    for block_i,block in enumerate(blocks):
        for repeat_i in range(block.num_repeats):
            name = 'block_%d/repeat_%d'%(block_i,repeat_i)
            
            conv = conv2d(net,block.bottleneck_size,k_h=1,k_w=1,
                          stride_h=1,stride_w=1,padding='VALID',
                          activation=activation,
                          name=name+'/conv_in')
            
            conv = conv2d(conv,block.bottleneck_size,k_h=3,k_w=3,
                          padding='SAME',stride_h=1,stride_w=1,
                          activation=activation,
                          name=name+'/conv_bottlneck')
            
            conv = conv2d(conv,block.num_filters,k_h=1,k_w=1,
                          padding='VALID',stride_h=1,stride_w=1,
                          activation=activation,
                          name=name+'/conv_out')
            net = conv + net
        
        try:
            next_block = blocks[block_i+1]
            net = conv2d(net,next_block.num_filters,k_h=1,k_w=1,
                         padding='SAME',stride_h=1,stride_w=1,bias=False,
                         name='block_%d/conv_upscale'%block_i)
        except IndexError:
            pass
    
    # Average Pool
    net = tf.nn.avg_pool(net,
                         ksize=[1,net.get_shape().as_list()[1],
                                net.get_shape().as_list()[2],1],
                         strides=[1,1,1,1],padding='VALID')
    net = tf.reshape(
        net,
        [-1, net.get_shape().as_list()[1] *
         net.get_shape().as_list()[2] *
         net.get_shape().as_list()[3]])
    
    # Fully Connection
    net = linear(net,n_output,activation=tf.nn.softmax)
    return net
    
    
def test_mnist():
    print('...... Building Model ......')
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])
    y_pred = residual_network(x,10)
    
    # Define loss and training functions
    cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    # Monitor Accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    
    # Session
    with tf.Session() as sess:
        print('...... Init Variable ......')
        init = tf.initialize_all_variables()
        sess.run(init)
        
        # Parameter
        batch_size = 50
        n_epochs = 5
        for epoch_i in range(n_epochs):
            # Training
            train_accuracy = 0
            for batch_i in range(mnist.train.num_examples // batch_size):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                train_accuracy += sess.run([optimizer,accuracy],feed_dict={
                                           x:batch_xs,y:batch_ys
                                           })[1]
            train_accuracy /= (mnist.train.num_examples//batch_size)
            
            # Validation
            valid_accuracy = 0
            for batch_i in range(mnist.validation.num_examples // batch_size):
                batch_xs,batch_ys = mnist.validation.net_batch(batch_size)
                valid_accuracy += sess.run(accuracy,
                                           feed_dict={
                                               x:batch_xs,
                                               y:batch_ys})
            valid_accuracy /= (mnist.validation.num_examples // batch_size)
            print('epoch:', epoch_i, ', train:',train_accuracy, ', valid:', valid_accuracy)
            
            
if __name__ == '__main__':
    test_mnist()