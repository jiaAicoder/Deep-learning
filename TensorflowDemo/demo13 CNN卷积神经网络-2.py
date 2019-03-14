#__author:"吉勇佳"
#date: 2019/3/13 0013
#function:

'''
图像的三层 RGB (256*256*1) --->>(128*128*16)--->(64*64*64) -->(32*32*256)

stride ：跨像素点抽离
patch ：小块的大小

'''

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1，是图片的厚度, out size 32
b_conv1 = bias_variable([32])
#搭建第一层(通过激活函数，进行非线性处理)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32 SAME模式长宽不变
#进行池化操作
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14x32  SAME模式下步长为2了，所以长宽缩小了一倍

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64，不断变厚
b_conv2 = bias_variable([64])              # 与输出是保持一致的。
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## func1 layer ## 最后的全连接层1
W_fc1 = weight_variable([7 * 7 * 64, 1024])    #输入是[7*7*64]  输出1024
b_fc1 = bias_variable([1024])                  #偏差是1024
# 把第二层输出的结果的三维信息变平，变为平面
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #输入与权重相乘加偏差
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout处理

## func2 layer ##最后的全连接层2
W_fc2 = weight_variable([1024, 10])    #因为我们需要判断的是0-9数字的分类，所以定义输出为10.
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)     #分类用softmax进行激活算概率


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   #这里一般不用梯度下降

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(4)    #这里batch不要设置太大，不然cpu受不了
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

