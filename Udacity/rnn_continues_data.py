# -*- coding: utf-8 -*-
"""
Using RNN models to predict continues Data like sine and cosine
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006

def get_batch():
    global BATCH_START,TIME_STEPS
    # xs shape()