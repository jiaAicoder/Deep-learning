
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
CNN步骤：
1）Image
2) Convolution
3) Max Pooling
4) Convolution
5) Max Pooling
6) Fully Connected
7) Fully Connected
8) Classification

本节课只完成Convolution层和pooling层。
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data   #手写识别数据
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)   #产生随机变量
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#1. Convolutional layer（卷积层--CONV）
'''
strides 是步长抽取，规则如下
padding 有两种，分别是valid 和same(抽取在外边，有一块是0填充)
'''
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#2. Pooling layer （池化层--POOL）(为了防止跨步太大，丢失太多，用pooling处理跨度大的问题)
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

## conv1 layer ##

## conv2 layer ##

## func1 layer ##

## func2 layer ##


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

'''
备查资料:(知乎链接:https://zhuanlan.zhihu.com/p/42559190)
三、CNN的结构组成
上面我们已经知道了卷积（convolution）、池化（pooling）以及填白（padding）是怎么进行的，接下来我们就来看看CNN的整体结构，它包含了3种层（layer）：
1. Convolutional layer（卷积层--CONV）
由滤波器filters和激活函数构成。 一般要设置的超参数包括f ilters的数量、大小、步长，以及padding是“valid”还是“same”。当然，还包括选择什么激活函数。
2. Pooling layer （池化层--POOL）
这里里面没有参数需要我们学习，因为这里里面的参数都是我们设置好了，要么是Maxpooling，要么是Averagepooling。 需要指定的超参数，包括是Max还是average，窗口大小以及步长。 通常，我们使用的比较多的是Maxpooling,而且一般取大小为(2,2)步长为2的filter，这样，经过pooling之后，输入的长宽都会缩小2倍，channels不变。
3. Fully Connected layer（全连接层--FC）
这个前面没有讲，是因为这个就是我们最熟悉的家伙，就是我们之前学的神经网络中的那种最普通的层，就是一排神经元。因为这一层是每一个单元都和前一层的每一个单元相连接，所以称之为“全连接”。 这里要指定的超参数，无非就是神经元的数量，以及激活函数。
'''