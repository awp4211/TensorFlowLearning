# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math


def lrelu(x,leak=0.2,name='lrelu'):
    """
    x:Tensor----to apply the nonlinearity to.
    leak:Float----Leakage parameter
    name:str,Varibale scopee to use
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


"""
Build a deep autoencoder
dimensions:list---Number of neurons for each layer of the autoencoder
"""        
def autoencoder(dimensions=[784,512,256,64]):
    x = tf.placeholder(tf.float32,[None,dimensions[0]],name='x')
    current_input = x
    
    print('...... Building the Forward AutoEncoder .......')
    encoder = []
    for layer_i,n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        w = tf.Variable(tf.random_uniform([n_input,n_output],
                                          -1.0 / math.sqrt(n_input),
                                           1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(w)
        output = tf.nn.tanh(tf.matmul(current_input,w) + b)
        current_input = output
    
    print('...... Successfully building Forward AutoEncoder ......')
    
    print('...... latent representation ......')
    z = current_input
    encoder.reverse()
    
    print('...... Build the decoder using the same weights ......')
    #reverse the autoencoder network size,(double dot reverse)
    for layer_i,n_output in enumerate(dimensions[:-1][::-1]):
        w = tf.transpose(encoder[layer_i])
        b = tf.zeros([n_output])
        output = tf.nn.tanh(tf.matmul(current_input,w)+b)
        current_input = output
        
    print('...... Successfulluy building reconstruction through the network ......')
    y = current_input
    
    cost = tf.reduce_sum(tf.square(y-x))
    return {'x':x,
            'z':z,
            'y':y,
            'cost':cost}

def test_mnist():
    print('...... Using the autoencoder using MNIST ......')
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt
    
    
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    
    # mean images
    mean_img = np.mean(mnist.train.images,axis=0)            
    ae = autoencoder(dimensions=[784,256,64])
    
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs,_ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer,feed_dict={ae['x']:train})
        print(epoch_i,sess.run(ae['cost'],feed_dict={ae['x']:train}))
        
    print('...... Plot example reconstructions ......')
    n_examples = 15
    test_xs,_ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'],feed_dict={ae['x']:test_xs_norm})
    fig,axs = plt.subplots(2,n_examples,figsize=(10,2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
            
if __name__ == '__main__':
    test_mnist()