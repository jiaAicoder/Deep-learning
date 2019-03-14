#__author:"吉勇佳"
#date: 2019/3/13 0013
#function:

'''
LSTM: long short time memory 长短期记忆
RNN: 循环神经网络
弊端：梯度消失和梯度爆炸，解决方案：long short time memory 长短期记忆
三个门：
'''
#import datasets
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data    #手写识别数据集

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

#read data
# mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)

#hyperparaments
lr = 0.001
training_items = 100000
batch_size = 4
# display_step = 10

# picture structure
n_inputs = 28              #img shape 28*28,so  this n_inputs is 28 points in row
n_steps = 28               #n step is 28 lines of a picture or img
n_hidden_unis = 128        #neurons in hidden units 即隐藏层神经元数
n_classes = 10             #how many classification

#tfGraph input of placeholder
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

#Define weights--在RNN 的两侧定义了两层的hidden_layer
weights = {
    # (28,128)
    "in":tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),  #input num and output number
    # (128,10)
    "out":tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}

biases = {
    #（128，）
    "in":tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    #(10,)
    "out":tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}


def RNN(X,weight,biases):
    #########################################
    #hidden layer for input to cell
    # X(128batch，28steps,28inputs)  --->(128*28,28 inputs)
    X = tf.reshape(X,[-1,n_inputs])     #变为连起来的二维,第一个参数-1即让电脑自己算
    X_in = tf.matmul(X,weight['in'])+biases['in']
    #（128batch，28steps,28inputs）
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])    #变回三维的形式，上一步二维计算，必须变为三维才可以传入下一步的cell


    #cell
    ##########################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    #Istm cell is devided into two parts(c_state,m_state)
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    #RNN caculate
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)   #time steps 在不在0 维度



    #hidden layer for putputs as final results
    ##########################################
    # 方法1
    # results = tf.matmul(states[1],weight['out'])+biases['out']   #state取的是分线剧情

    # 方法2：取output[-1].
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)



    return results


#get the prediction and caculate the cost and accuracy
pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)   #减小误差

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


#initial variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size<training_items:  #终止条件
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })

        if step%20 ==0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys
            }))
            step += 1

