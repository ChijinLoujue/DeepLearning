# 卷积网络和池化内容  在MNIST进阶中
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("D:/WorkSpace/GitHub/DeepLearning/Data/MNIST_data/", one_hot=True)
 
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

x = tf.placeholder("float", [None, 784])  # 设置一个占位符x，
y_ = tf.placeholder("float", [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#####定义卷积
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')   #1步长，0边距
#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
#  除去name参数用以指定该操作的name，与方法有关的一共五个参数：
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
# 具有[batch, in_height, in_width, in_channels]这样的shape，
# 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
# 注意这是一个4维的Tensor，要求类型为float32和float64其中之一
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
# 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
# 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
# 要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
# 结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。


#####定义池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


#这是第一层：一个卷积＋池化
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 二层：卷积＋池化
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
sess = tf.InteractiveSession()
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
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
