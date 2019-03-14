#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:


import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #总结：用了placeholder的话就在run 的时候再传入参数，字典的形式feed_dict
    print(sess.run(output,feed_dict={input1:[7.0],input2:[2.0]}))

