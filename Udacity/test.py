# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:16:37 2016

@author: zc
"""

import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])

matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run([product])
    print result