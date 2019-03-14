#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:
'''
bug修改：1、writer = tf.train.SummaryWriter("logs/", sess.graph) 改为：
writer = tf.summary.FileWriter("logs/", sess.graph)
【重要】：执行 tensorboard时候的错误：
1、必须是一个等号，为 tensorboard --logdir="C:\logs"
2、必须是双引号
3、cmd是站在logs母文件夹的体内的，是不进logs文件夹的。
'''

'''
结构：input-layer --->layer[隐藏层1] --->layer-1[隐藏层:2] --->loss ---train
layer-1：weights,biases,wx-plub_b
input-layer:input-x,input-y
layer:weights,biases,wx-plub_b,Rule

'''

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
log_dir= r'.\logs'
writer = tf.summary.FileWriter(log_dir, sess.graph)
# important step
sess.run(tf.initialize_all_variables())


'''
如何查看收成的图：
1、进入terminal.进入项目logs文件夹所在的目录
2、执行：tensorboard --logdir='logs/'
3、copy 返回的网址信息
'''

