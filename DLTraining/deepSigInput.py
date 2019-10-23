from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import cmath
#import plotly.plotly as py

#数据集步进为  106495
DATA_FILE_NAME = "D:/Study/ModulationRecognition/MyCode/MRpy/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

def dsdata_input():
    deepsigData = h5py.File(DATA_FILE_NAME, 'r')
    datakey = list(deepsigData.keys())
    print(datakey)
    X = deepsigData[datakey[0]]
    Y = deepsigData[datakey[1]]
    Z = deepsigData[datakey[2]]
    print('X=', X)
    print('Y=', Y)
    print('Z=', Z)
    print('Finish loading.')
    LENGTHofDATA = len(X[:, 1, 1])
    dsdata = tf.data.Dataset.from_tensor_slices(X, Y, Z)
    dsdata = dsdata.shuffle(LENGTHofDATA).batch(1000)
    return dsdata



