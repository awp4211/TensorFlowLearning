# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import process_data as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt


n_train_example = 10340
n_test_example = 1449

learning_rate = 0.001
batch_size = 1000
n_class = 2


"""
Take an input tensor and add uniform masking
x:tensor/placeholder
x_corrupt:50% of value corrupt
"""
def corrupt(x):
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

"""
Build a deep denoising autoencoder
parameters:
    width,height:input dimension
    dimensions:list of DAE dimension
"""
def autoencoder(x,
                corrupt_prob,
                width=256,height=256,
                dimensions=[784,512,256,64]):
    # current layer input
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)
    curretn_input = x
    
    dimensions.insert(0,width*height)    
    
    print('...... building the encoder ......')
    encoder = []
    for layer_i,n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape().as_list()[1])
        W = tf.Variable(
                tf.random_uniform([n_input,n_output],
                                  -1.0/math.sqrt(n_input),
                                   1.0/math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input,W)+b)
        current_input = output
        print('DAE forward,layer_{0},n_input={1},n_output={2}'.format(layer_i,n_input,n_output))
    
    # latent representation
    z = current_input

    print('...... building the decoder ......')    
    
    encoder.reverse()
    for layer_i,n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output,]))
        output = tf.matmul(current_input,W)
        output = tf.add(output,b)
        output = tf.nn.tanh(output)
        print('DAE backward,layer_{0},n_input={1},n_output={2}'.format(layer_i,W.get_shape()[0],n_output))
        current_input = output
        
    # now have the recontruction through the network
    y = current_input    
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square( y - x)))
    return {'x':x,'z':z,'y':y,
            'corrupt_prob':corrupt_prob,
            'cost':cost}
            
"""
inputs:autoencoder-z
"""
def classify(inputs):
    n_input = inputs.get_shape().as_list()[1]
    
    weights = tf.Variable(tf.random_normal([n_input,n_class]))
    biases = tf.Variable(tf.constant(0.1,shape=[n_class,]))
    
    result = tf.matmul(inputs,weights) + biases
    result = tf.nn.softmax(result)
    return result

    
def train_yawn(width=256,height=256):
    
    d_start = datetime.datetime.now()
    
    print('...... loading the dataset ......')
    train_set_x,train_set_y,test_set_x,test_set_y = pd.load_data_set(width,height)
    
    train_mean = np.mean(train_set_x,axis=0)
    test_mean = np.mean(test_set_x,axis=0)
    
    x = tf.placeholder(tf.float32,[None,width*height]) # input
    y = tf.placeholder(tf.float32,[None,n_class])    # label
    corrupt_prob = tf.placeholder(tf.float32,[1])

    ae = autoencoder(x,corrupt_prob,
                     width,height,
                     dimensions=[5000,300,
                                 200])
    y_pred = classify(ae['z'])
    print('y_pred,shape = {}'.format(y_pred.get_shape()))
    
    cost = ae['cost']
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    best_acc = 0.
    
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        print('...... initializating varibale ...... ')
        sess.run(init)
        
        n_epochs = 5
        print('...... start to training ......')
        for epoch_i in range(n_epochs):
            # Training 
            train_accuracy = 0.
            for batch_i in range(n_train_example//batch_size):
                
                batch_xs = train_set_x[batch_i*batch_size:(batch_i+1)*batch_size]
                batch_xs_norm = np.array([img - train_mean for img in batch_xs])
                batch_ys = train_set_y[batch_i*batch_size:(batch_i+1)*batch_size]
                _,loss,acc = sess.run([optimizer,cost,accuracy],
                                           feed_dict={
                                                x:batch_xs_norm,
                                                y:batch_ys,
                                                corrupt_prob:[1.0]}
                                                )
                #print('epoch:{0},minibatch:{1},y_res:{2}'.format(epoch_i,batch_i,yy_res))
                #print('epoch:{0},minibatch:{1},y_pred:{2}'.format(epoch_i,batch_i,yy_pred))
                print('epoch:{0},minibatch:{1},cost:{2},train_accuracy:{3}'.format(epoch_i,batch_i,loss,acc))
                train_accuracy += acc

            train_accuracy /= (n_train_example//batch_size)
            print('----epoch:{0},training acc = {1}'.format(epoch_i,train_accuracy))
            
            # Validation
            valid_accuracy = 0.
            for batch_i in range(n_test_example//batch_size):
                batch_xs = test_set_x[batch_i*batch_size:(batch_i+1)*batch_size]
                batch_xs_norm = np.array([img - test_mean for img in batch_xs])
                batch_ys = test_set_y[batch_i*batch_size:(batch_i+1)*batch_size]
                valid_accuracy += sess.run(accuracy,
                                           feed_dict={
                                                x:batch_xs,
                                                y:batch_ys,
                                                corrupt_prob:[0.0]})
            valid_accuracy /= (n_test_example//batch_size)
            print('epoch:{0},train_accuracy:{1},valid_accuracy:{2}'.format(epoch_i,train_accuracy,valid_accuracy))
            if(train_accuracy > best_acc):
                best_acc = train_accuracy
         # draw
        n_examples = 10
        test_xs = test_set_x[n_examples*130:n_examples*131]
        test_xs_norm = np.array([img - train_mean for img in test_xs])
        recon = sess.run(ae['y'],feed_dict={x:test_xs_norm,
                                             corrupt_prob:[0.0]})
        fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (width, height)))
            axs[1][example_i].imshow(
                np.reshape([recon[example_i, :] + train_mean], (width, height)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()
        
    d_end = datetime.datetime.now()
    print('...... training finished ......')
    print('...... best accuracy:{0} ......'.format(best_acc))
    print('...... running time:{0} minutes ......'.format( (d_end-d_start).seconds/60))
    
if __name__ == '__main__':
    if len(sys.argv) == 3:
        # python *.py width height
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        print('...... training DAE customer size:width={0},height={1}'.format(w,h))
        train_yawn(w,h)
    else:
        #len(sys.argv) == 1 python *.py
        print('...... training DAE default size:width={0},height={1}'.format(256,256))
        train_yawn()
        pass
    