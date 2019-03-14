#__author:"吉勇佳"
#date: 2019/3/13 0013
#function:

# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 80

n_inputs = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    # transpose the inputs shape from
    # X (80,28,28) ==> (80 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (80 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (80 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 基本的LSTM Cell. 有：
    # BasicLSTMCell & BasicRNNCell & GRUCell & LSTMCell & RNNCell
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)  # 128
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # 下一步有两个选择
    # 1: tf.nn.static_rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # 方法1
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

    # 方法2
    # dynamic_rnn的输入X_in为 (batch, steps, inputs) 或 (steps, batch, inputs)
    # 这里为 (batch, steps, inputs)
    # time_major的意思是：是否steps为第一个参数，这里不是，则false.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    #############################################
    # print(sess.run(tf.shape(outputs)))      #[ 80  28 128]
    # print(sess.run(tf.shape(final_state)))  # [ 2  80 128]
    # final_state包括(c_state, h_state)长线记忆和短线记忆
    #############################################

    # hidden layer for output as the final results
    # 第一种方法
    # final_state为[2,80,128]     则final_state[1]为[80,128]
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # 第二种方法
    # outputs = tf.transpose(outputs, [1,0,2])   这条语句也可以的
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys, })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, }))
        step += 1
