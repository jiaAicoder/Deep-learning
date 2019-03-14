#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:

import tensorflow as tf
import numpy as np


# 添加一个神经层---函数
def add_layer(input, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 生成in_size * out_size 的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    '''
    [[0. 0. 0. 0. 0.]]  tensorflow 中规定了biases是不为零的
    [[0.1 0.1 0.1 0.1 0.1]]   所以加上一个数

    '''
    Wx_plus_y = tf.matmul(input, Weights) + biases  # 即是一个预测输出的y值

    if activation_function is None:
        outputs = Wx_plus_y  # 即没有激活函数的话保持线性关系
    else:
        outputs = activation_function(Wx_plus_y)

    return outputs


#输入的信息数据
x_s = tf.placeholder(tf.float32,[None,1])    #先暂时holder住，run的时候再传入值,None 是代表shape,1 代表1个输入
y_s = tf.placeholder(tf.float32,[None,1])

x_data = np.linspace(-1,1,300)[:,np.newaxis]    #numpy.ndarray,加入一个维度转为矩阵
noise = np.random.normal(0,0.05,x_data.shape)    #0,0.05 是均值和方差
y_data = np.square(x_data) - 0.5+noise

'''
输入层:有多少个特征就有多少个神经元
隐藏层:可以自己预设
输出层:一个神经元
'''

#造层1 -隐藏层-_layer
l1 = add_layer(x_s,1,10,activation_function=tf.nn.relu)    #1,10  1代表是1个输入，10代表是10 个输出
#造层2 --输出层 -out-layer
prediction = add_layer(l1,10,1,activation_function=None)

#loss值计算
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_s-prediction),reduction_indices=[1]))

#train——step
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #作用是减少误差    #学习率0.1

#所以的变量初始化
init=tf.initialize_all_variables()

#入口
sess = tf.Session()
sess.run(init)
for i in range(50000):
    sess.run(train_step,feed_dict={x_s:x_data,y_s:y_data})
    if i % 50:

        print(sess.run(loss,feed_dict={x_s:x_data,y_s:y_data}))   #凡是涉及到计算的地方都要传入feed_dict

