# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import sys

clazz_count = 2

def process_data(width=256,height=256):
    _train_file = 'YawnData/train.txt'
    _test_file = 'YawnData/test.txt'
    
    train_file = open(_train_file,'rt')
    test_file = open(_test_file,'rt')
    
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    print('...... processing training set......')
    index = 0
    for line in train_file.readlines():
        line = line.strip()
        img_path = 'YawnData/' + line.split()[0]
        label = line.split()[1]
        vec_img = convert_image_to_vec(img_path,width,height)
        vec_label = one_hot_encoder(int(label),clazz_count)
        train_set_x.append(vec_img)
        train_set_y.append(vec_label)
        index += 1
        if index%100 == 0:
            print('processed {0} images'.format(index))
    
    train_set_x = np.asarray(train_set_x)
    train_set_y = np.asarray(train_set_y)
    file1 = 'Yawn_dataset/train_set_x_{0}_{1}.npz'.format(width,height)
    file2 = 'Yawn_dataset/train_set_y_{0}_{1}.npz'.format(width,height)
    np.save(file1,train_set_x)
    np.save(file2,train_set_y)
    
    print('...... processing test set......')
    index = 0
    for line in test_file.readlines():
        line = line.strip()
        img_path = 'YawnData/' + line.split()[0]
        label = line.split()[1]
        vec_img = convert_image_to_vec(img_path,width,height)
        vec_label = one_hot_encoder(int(label),clazz_count)
        test_set_x.append(vec_img)
        test_set_y.append(vec_label)
        index += 1
        if index%100 == 0:
            print('processed {0} images'.format(index))
    
    test_set_x = np.asarray(test_set_x)
    test_set_y = np.asarray(test_set_y)
    file3 = 'Yawn_dataset/test_set_x_{0}_{1}.npz'.format(width,height)
    file4 = 'Yawn_dataset/test_set_y_{0}_{1}.npz'.format(width,height)
    np.save(file3,test_set_x)
    np.save(file4,test_set_y)
        
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
        
def one_hot_encoder(label,class_count):
    mat = np.asarray(np.zeros(class_count),dtype='float64').reshape(1,class_count)
    for i in range(class_count):
        if i == label:
            mat[0][i] = 1
    return mat.flatten()
    
def load_data_set(width=256,height=256):
    file1 = 'Yawn_dataset/test_set_x_{0}_{1}.npz.npy'.format(width,height)
    file2 = 'Yawn_dataset/test_set_y_{0}_{1}.npz.npy'.format(width,height)
    file3 = 'Yawn_dataset/train_set_x_{0}_{1}.npz.npy'.format(width,height)
    file4 = 'Yawn_dataset/train_set_y_{0}_{1}.npz.npy'.format(width,height)
    
    test_set_x = np.load(file1)
    print('...... Extracted test_set_x file:{0} ......'.format(file1))
    test_set_y = np.load(file2)
    print('...... Extracted test_set_y file:{0} ......'.format(file2))    
    train_set_x = np.load(file3)
    print('...... Extracted train_set_x file:{0} ......'.format(file3))
    train_set_y = np.load(file4)
    print('...... Extracted train_set_y file:{0} ......'.format(file4))
    
    return train_set_x,train_set_y,test_set_x,test_set_y
    
if __name__ == '__main__':
    if len(sys.argv) == 3:
        print('...... process_data customer:width = {0},height = {1}'.format(sys.argv[1],sys.argv[2]))
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        process_data(width=w,height=h)
    else:
        print('...... process_data default:width = 256,height = 256')
        process_data(width=256,height=256)