#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:

import tensorflow as tf
biases = tf.Variable(tf.zeros([1,5])+ 0.1 )
biases1 = tf.Variable(tf.zeros([1,5]) )

init = tf.initialize_all_variables()    #must have if define variable\

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(biases))
    print(sess.run(biases1))
