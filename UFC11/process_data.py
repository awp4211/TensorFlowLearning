# -*- coding: utf-8 -*-
import cPickle
import PIL
from PIL import Image

import numpy as np

clazz_count = 11

def process_data(width=256,height=256):
    
    _train_path_file = 'UFC11/_train_file.txt'
    _train_label_file = 'UFC11/_train_label.txt'
    _test_path_file = 'UFC11/_test_file.txt'
    _test_label_file = 'UFC11/_test_label.txt'
    
    train_file = open(_train_path_file,'rt')
    train_label = open(_train_label_file,'rt')
    test_file = open(_test_path_file,'rt')
    test_label = open(_test_label_file,'rt')

    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []     
    
    index = 0
    #===testing set 4872 = 24 * 203
    for img_path in test_file.readlines():
        img_path = img_path[0:-2]#去掉/r/n
        vec = convert_image_to_vec(img_path,width,height)
        test_set_x.append(vec)
        index += 1
        if index%100 == 0:
            print('processed {0} images'.format(index))

    test_set_x = np.asarray(test_set_x)
    
    #===testing set 203
    for label in test_label.readlines():
        vec = one_hot_encoder(label,clazz_count)
        test_set_y.append(vec)    
    
    test_set_y = np.asarray(test_set_y)
    
    index = 0
    #===training set 33528 = 24 * 1397
    for img_path in train_file.readlines():
        img_path = img_path[0:-2]#去掉/r/n
        vec = convert_image_to_vec(img_path,width,height)
        train_set_x.append(vec)
        index += 1
        if index%100 == 0:
            print('processed {0} images'.format(index))
            
    train_set_x = np.asarray(train_set_x)
    
    #===training label 1397
    for label in train_label.readlines():
        vec = one_hot_encoder(label,clazz_count)
        train_set_y.append(vec)
    
    train_set_y = np.asarray(train_set_y)
    
    print(type(train_set_x))    
    print(type(train_set_y))
    print(type(test_set_x))
    print(type(test_set_y))

    print(train_set_x.shape)  
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)    

    # 保存到pickle
    file_name = 'ufc11_{0}_{1}.pkl.gz'.format(width,height)
    f = open(file_name,'wb')
    cPickle.dump(train_set_x,f,-1)
    cPickle.dump(train_set_y,f,-1)   
    cPickle.dump(test_set_x,f,-1)
    cPickle.dump(test_set_y,f,-1)    
    
  
def one_hot_encoder(label,class_count):
    mat = np.asarray(np.zeros(class_count),dtype='float64').reshape(1,class_count)
    for i in range(class_count):
        if i == label:
            mat[0][i] = 1
    return mat.flatten()
        

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
    process_data(width=256,height=256)