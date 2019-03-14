#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:

'''
结构：input-layer --->layer[隐藏层1] --->layer-1[隐藏层:2] --->loss ---train
input-layer:input-x,input-y
layer:weights,biases,wx-plub_b,Rule
layer-1：weights,biases,wx-plub_b


'''

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#Create tensorflow structure
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))    #[1] 一维结构 从-1 到1
biases = tf.Variable(tf.zeros([1]))     #初始值是0
y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)     #优化器 0.5 是学习效率
train = optimizer.minimize(loss)

init= tf.initialize_all_variables()             #初始化变量
#end creat

###激活###
sess = tf.Session()
sess.run(init)

#训练#
for step in range(3000):
    sess.run(train)
    if step % 20==0:
        print(step,sess.run(Weights),sess.run(biases))

