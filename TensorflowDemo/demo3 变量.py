#__author:"吉勇佳"
#date: 2019/3/10 0010
#function:


import tensorflow as tf

#变量 variable
state = tf.Variable(0,name='counter')
print(state.name)

#常量 constant
one = tf.constant(1)

new_value = tf.add(state,one)     #每次加1
update = tf.assign(state,new_value)

#初始化变量和run才能激活
init = tf.initialize_all_variables()    #must have if define variable

#入口
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

