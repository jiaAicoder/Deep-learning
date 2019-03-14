#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)        #矩阵的乘法，同np.dot(m1,m2)

#method1  --the way1 to open
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

#method2  --the way2 to open [introduct]
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
