#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:


'''

上节--激励函数
'''
import tensorflow as tf
import numpy as np

#添加一个神经层---函数
def add_layer(input,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))   #生成in_size * out_size 的矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+ 0.1 )
    '''
    [[0. 0. 0. 0. 0.]]  tensorflow 中规定了biases是不为零的
    [[0.1 0.1 0.1 0.1 0.1]]   所以加上一个数
    
    '''
    Wx_plus_y = tf.matmul(input,Weights)+biases    #即是一个预测输出的y值

    if activation_function is None:
        outputs = Wx_plus_y   #即没有激活函数的话保持线性关系
    else:
        outputs = activation_function(Wx_plus_y)

    return outputs
