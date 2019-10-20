#这个应该是之前官网教程基础那个
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trainimg = mnist.train.images     #定义导入的数据 ，分为训练集和测试集
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

x = tf.placeholder("float", [None, 784])             #设置一个占位符x，
y_ = tf.placeholder("float", [None,10])              #设置一个占位符y_
W = tf.Variable(tf.zeros([784,10]))                #定义权值变量W,784×10
b = tf.Variable(tf.zeros([10]))              #定义偏置量b,10个

sess = tf.InteractiveSession()             #   定义一个交互式计算图
init = tf.initialize_all_variables()       #   初始化所有变量
sess.run(init)                             #  让初始化再这个图里跑起来

y = tf.nn.softmax(tf.matmul(x,W) + b)      #  输出y为softmax函数网络

learning_rate = 0.01                       #  设学习率为0.01
cross_entropy = -tf.reduce_sum(y_*tf.log(y))   # 交叉熵
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
                  #   设计学习步进为学习率，方法为交叉熵的最小梯度
training_epochs = 50        #训练迭代为50
batch_size = 10              
num_batch = int(mnist.train.num_examples/batch_size)

for i in range(num_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
