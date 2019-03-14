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

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer,activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)     #需要看变量变化的话加一个这个
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)

        return outputs

#make up some new data
x_data = np.linspace(-1,1,300)[:,np.newaxis]    #numpy.ndarray,加入一个维度转为矩阵
noise = np.random.normal(0,0.05,x_data.shape)    #0,0.05 是均值和方差
y_data = np.square(x_data) - 0.5 + noise


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2,activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss',loss)                  #观察loss的语句与上述不同


with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
#打包
merged = tf.summary.merge_all()

log_dir= r'.\logs1'
writer = tf.summary.FileWriter(log_dir, sess.graph)
# important step
sess.run(tf.initialize_all_variables())

#入口
for i in range(10000):
    #训练
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)


'''
如何查看收成的图：
1、进入terminal.进入项目logs文件夹所在的目录
2、执行：tensorboard --logdir='logs/'
3、copy 返回的网址信息
'''


'''
AttributeError: 'module' object has no attribute 'SummaryWriter'

tf.train.SummaryWriter

改为：tf.summary.FileWriter

AttributeError: 'module' object has no attribute 'summaries'

tf.merge_all_summaries()

改为：summary_op = tf.summary.merge_all()

AttributeError: 'module' object has no attribute 'histogram_summary'

tf.histogram_summary()

改为：tf.summary.histogram()

tf.scalar_summary()

改为：tf.summary.scalar()

tf.image_summary()

改为：tf.summary.image()

参考：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md


参考：http://blog.csdn.net/edwards_june/article/details/65652385
--------------------- 
作者：激进的小鸡蛋 
来源：CSDN 
原文：https://blog.csdn.net/waterydd/article/details/70237984 

'''