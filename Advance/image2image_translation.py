# -*- coding: utf-8 -*-

"""
This is the implementation of pix2pix
paper:Image-to-Image Translation with Conditional Adversarial Networks
"""

import tensorflow as tf
import numpy as np
import os
import json
import glob
import random
import collections
import time
import math

EPS = 1e-12
CROP_SIZE = 256

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def download_dataset():
    """
    Downloading facades dataset
    :return:
    """
    import sys
    import tarfile
    import tempfile
    import shutil
    from urllib2 import urlopen  # python 2
    url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % 'facades'
    with tempfile.TemporaryFile() as tmp:
        print("downloading", url)
        shutil.copyfileobj(urlopen(url), tmp)
        print("extracting")
        tmp.seek(0)
        tar = tarfile.open(fileobj=tmp)
        tar.extractall()
        tar.close()
        print("done")

