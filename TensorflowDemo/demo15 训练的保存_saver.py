import tensorflow as tf
import numpy as np

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

'''
#方法1


## Save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init= tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    save_path1 = r'./files/save_net.ckpt'
    sess.run(init)
    save_path = saver.save(sess, save_path1)
    print("Save to path: ", save_path)



'''
#方法2
################################################
# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# W = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float32, name="weights")
# b = tf.Variable([[7,8,9]], dtype=tf.float32, name="biases")
# 

# initial the viaiable
init = tf.initialize_all_variables()    #must have if define variable

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    save_path = r'./files/save_net1.ckpt'
    saver.restore(sess, save_path)
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))














