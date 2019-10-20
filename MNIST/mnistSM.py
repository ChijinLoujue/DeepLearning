#这应该是之前视频教学里的那个格式，可以测试tensorflow用
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
#inport this package,which is in tesnorflow
#needn't download this input_data.py in internet it still exists in tensorflow

print("packs loads")
print("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#download the data file to "MNIST_data",a new subfolder in project folder
#read the data to the mnist
print("type of 'mnist' is ", type(mnist))
print("number of train data is %d" %(mnist.train.num_examples))
print("number of test data is %d" %(mnist.test.num_examples))
#let's see some information about input data
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
#read the kinds of data to different variable
print("shape of 'trainimg' is ", trainimg.shape)
print("shape of 'trainlabel' is ", trainlabel.shape)
print("shape of 'testimg' is ", testimg.shape)
print("shape of 'testlabel' is ", testlabel.shape)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None,10])
#set x and y_ as placeholders，None means any long data with uncertain length length
W = tf.Variable(tf.zeros([784,10]))                #定义权值变量W,784×10
b = tf.Variable(tf.zeros([10]))              #定义偏置量b,10个

#logic regression mode
y = tf.nn.softmax(tf.matmul(x,W) + b)         #一个softmax函数，matmul矩阵相乘，softmax柔性最大值传输函数
# 有限项离散概率分布的梯度对数归一化
learning_rate = 0.01
cross_entropy = -tf.reduce_sum(y_*tf.math.log(y))     #交叉熵
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy) #梯度下降法求解最小交叉熵
#梯度下降优化器GradientDescentOptimizer
#上面两行为直接用交叉熵计算，不需要进行

init = tf.initialize_all_variables()       #初始化所有变量

sess = tf.Session()
sess.run(init)

training_epochs = 50
batch_size = 100
display_step = 5

num_batch = int(mnist.train.num_examples/batch_size)
for i in range(num_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#feed_dict是为了给x和y_两个占位符赋值，

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 最大值索引，参数为0，寻找列最大值下标；参数为1，寻找行最大下标
#对比计算值和真实值对比，一致则为true，不一致则为false

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#tf.cast用于把boll型转为float型，即1和0两个数值
#reduce_mean求均值，即求正确率

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
