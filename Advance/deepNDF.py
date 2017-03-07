# -*- coding: utf-8 -*-
"""
This is a tutorial implemention of Deep Neural Decision Forests
(dNDF).The paper was published in ICCV(the Open Access vision,provided by
the Computer Vision Foundation).

Deep Neural Decision Forest,ICCV 2015,proposed a great way to incorporate a 
neural network with a decision forest.Durning the optimization,the terminal
(leaf)node has to be updated after each epoch.

This code tested a simple(3 convolution + 2 FC) network for the experiment
"""

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

#parameter setting
DEPTH = 3                  # Depth of a tree
N_LEAF = 2** (DEPTH +1)    # Number of leaf node
N_LABEL = 10               # Number of classes
N_TREE = 5                 # Number of trees(ensemble)
N_BATCH = 128              # Number of data points per mini-batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def init_prob_weights(shape,minval=-5,maxval=5):
    return tf.Variable(tf.random_normal(shape,minval,maxval))
    
def model(X,
          w,
          w2,
          w3,
          w4_e,
          w_d_e,
          w_l_e,
          p_keep_conv,
          p_keep_hidden):
    """
    Create a forest and return the neural decision forest outputs:
        decision_p_e:decision node routing probability for all ensemble
            if we number all nodes in the tree sequentially from top to bottom,
            left to right,decision_p contains
            [d(0),d(1)....d(2^n-2)]
            
            decision_p_e is the concatenation of all tree decision_p
            
        leaf_p_e:terminal node probability distributions for all ensemble.
            The indexing is the same as that of decision_p_e
    """
    
    assert(len(w4_e) == len(w_d_e))
    assert(len(w4_e) == len(w_l_e))
    
    l1a = tf.nn.relu(tf.nn.conv2d(input=X,
                                  filter=w,
                                  strides=[1,1,1,1],
                                  padding='SAME'))
    l1 = tf.nn.max_pool(l1a,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME')

    l1 = tf.nn.dropout(l1,p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1,
                                  filter=w2,
                                  strides=[1,1,1,1],
                                  padding='SAME'))
    l2 = tf.nn.max_pool(l2a,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME')
    l2 = tf.nn.dropout(l2,p_keep_conv)
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2,
                                  filter=w3,
                                  strides=[1,1,1,1],
                                  padding='SAME'))
    l3 = tf.nn.max_pool(l3a,ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
    l3 = tf.reshape(l3,[-1.,w4_e[0].get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3,p_keep_conv)
    
    decision_p_e = []
    leaf_p_e = []
    for w4,w_d,w_l in zip(w4_e,w_d_e,w_l_e):
        l4 = tf.nn.relu(tf.matmul(l3,w4))
        l4 = tf.nn.dropout(l4,p_keep_hidden)
        
        decision_p = tf.nn.sigmoid(tf.matmul(l4,w_d))
        leaf_p = tf.nn.softmax(w_l)
        
        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)
        
    return decision_p_e,leaf_p_e
    
def train():
    # load dataset
    print('...... loading dataset ......')
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    trX, trY = mnist.train.images,mnist.train.labels
    teX, teY = mnist.test.images,mnist.test.labels
    trX = trX.reshape(-1,28,28,1)
    teX = teX.reshape(-1,28,28,1)
    
    print('...... initializting ......')
    X = tf.placeholder(tf.float32,[N_BATCH,28,28,1])
    Y = tf.placeholder(tf.float32,[N_BATCH,N_LABEL])
    
    w = init_weights([3,3,1,32])
    w2 = init_weights([3,3,32,64])
    w3 = init_weights([3,3,64,128])
    
    w4_ensemble = []
    w_d_ensemble = []
    w_l_ensemble = []
    for i in range(N_TREE):
        w4_ensemble.append(init_weights([128*4*4,625]))
        w_d_ensemble.append(init_prob_weights([625,N_LEAF],-1,1))
        w_l_ensemble.append(init_prob_weights([N_LEAF,N_LABEL],-2,2))
    
    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)
    
    decision_p_e ,leaf_p_e = model(X,w,w2,w3,w4_ensemble,w_d_ensemble,
                                   w_l_ensemble,p_keep_conv,p_keep_hidden)
    flat_decision_p_e = []
    
    for decision_p in decision_p_e:
        decision_p_comp = tf.sub(tf.ones_like(decision_p),decision_p)
        
        decision_p_pack = tf.pack([decision_p,decision_p_comp])
        
        flat_decision_p = tf.reshape(decision_p_pack.[-1])
        flat_decision_p_e.append(flat_decision_p)
        
    
    
    
                