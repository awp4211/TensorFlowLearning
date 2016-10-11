# -*- coding: utf-8 -*-
import cPickle
import PIL
from PIL import Image

import numpy as np

def process_data(width=256,height=256):
    
    _train_path_file = 'UFC11/_train_file.txt'
    _train_label_file = 'UFC11/_train_label.txt'
    _test_path_file = 'UFC11/_test_file.txt'
    _test_label_file = 'UFC11/_test_label.txt'
    
    train_file = open(_train_path_file,'rt')
    train_label = open(_train_label_file,'rt')
    test_file = open(_test_path_file,'rt')
    test_label = open(_test_label_file,'rt')
    
    #===training set 33528 = 24 * 1397
    for img_path in train_file.readlines():
        pass
    
    #===training label 1397
    for label in train_label.readlines():
        pass        
     
    #===testing set 4872 = 24 * 203
    for img_path in test_file.readlines():
        pass
    
    #===testing set 203
    for label in test_label.readlines():
        pass
    

def convert_image_to_vec(img,width,height):
    im = Image.open(img)
    im = im.convert('L')
    mat = np.asarray(np.arange(width*height),dtype='float32').reshape(width,height)
    im = im.resize((width,height),Image.ANTIALIAS)  
    for i in range(0,list(im.size)[0]):
        for j in range(0,list(im.size)[1]):
            mat[i][j] = float(im.getpixel((i,j)))/255.0
    #使用flatten之后变成一维向量
    return mat.reshape(1,width*height).flatten()

def load_data_set():
    pass
        
if __name__ == '__main__':
    process_data()