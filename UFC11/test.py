# -*- coding: utf-8 -*-
import cPickle
import PIL
from PIL import Image

import numpy as np

def process_data(width=256,height=256):
    img = '/home/zc/Desktop/pic.png'
    im = Image.open(img)
    im = im.convert('L')
    mat = np.asarray(np.arange(width*height),dtype='float32').reshape(width,height)
    im = im.resize((width,height),Image.ANTIALIAS)
    pix = im.load()
    
    
    for i in range(width):
        for j in range(height):
            mat[i][j] = pix[i,j] /255.0
    return mat

if __name__ == '__main__':
    print(process_data())