# -*- coding: utf-8 -*-

"""
Author:XiyouZhaoC
Going Deeper with Convolution
    InceptionV1 and LSTM networks
    

UFC11 dataset contains 1600 videos and hava been classified 11 classes 
"""
import tensorflow as tf
import sys
import process_data as pd

# Dataset count
n_train_example = 33528
n_test_example = 4872

# Network Parameter
learning_rate = 0.001
dropout_keep_prob = 0.8

pic_batch_size = 2400 # % fps == 0
fps = 24
video_batch_size = pic_batch_size / fps
n_classes = 11


# LSTM Parameter
n_hidden_units = 384

