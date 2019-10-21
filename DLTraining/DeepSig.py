# 卷积网络和池化内容  在MNIST进阶中
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
# import tensorflow.examples.tutorials.mnist.input_data as input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
# trainimg = mnist.train.images
# trainlabel = mnist.train.labels
# testimg = mnist.test.images
# testlabel = mnist.test.labels
deepsigData = h5py.File('D:/Study/ModulationRecognition/MyCode/MRpy/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
datakey = list(deepsigData.keys())
print(datakey)
X = deepsigData[datakey[0]]
print('X=', X)
Y = deepsigData[datakey[1]]
print('Y=', Y)
Z = deepsigData[datakey[2]]
print('Z=', Z)







# Ximag = X[0, 0:1023, 0]
# Xreal = X[0, 0:1023, 1]
#
# plt.close()
# plt.figure(0)
# plt.plot(Ximag, Xreal, '.r')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
#
plt.figure(1)
Y0 = Y[1555904, 0:23]
YPart = Y[0:5904, 0:23]
index11 = tf.argmax(YPart, 0)
plt.plot(YPart[0:100, 0])
plt.show()
#
# Z0 = Z[0, 0]
# print(Z0)

"""

x = tf.placeholder("float", [None, 1024])  # 设置一个占位符x，
y_ = tf.placeholder("float", [None, 24])
#####定义权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#####定义偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#####定义卷积
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')   #1步长，0边距
#####定义池化
def max_pool_2x1(x):
    return tf.nn.pool(x,ksize=[1,2,1,1],
                      strides=[1,2,1,1],padding='SAME')

#这是第一层：一个卷积＋池化
w_conv1 = weight_variable([5,5,1,64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x,[-1,1024,2,1])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,64,1024,1],strides=[1,2,2,1],padding='SAME')

#第二层：卷积＋池化
w_conv2 = weight_variable([5,5,64,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)



sess = tf.InteractiveSession()  #
init = tf.initialize_all_variables()
sess.run(init)


learning_rate = 1e-4
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables)

training_epochs = 50
batch_size = 10
num_batch = int(mnist.train.num_examples / batch_size)

for i in range(num_batch):
    batch = mnist.train.next_batch(training_epochs)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print ("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
            x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
"""